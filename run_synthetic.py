import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def entropy_loss(x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = -1.0 * b.sum()
    
    return b

def train(i_epoch, 
          filter, optimizer_filter,
          filter_discriminator, optimizer_filter_discriminator,
          generator, optimizer_generator,
          generator_discriminator, optimizer_generator_discriminator, 
          dataloader, opt):
    """ Runs one training epoch. """
    # set train mode
    filter.train()
    filter_discriminator.train()
    generator.train()
    generator_discriminator.train()

    # ---------------------
    #  Train Discriminator
    # ---------------------
#     for i_batch, batch in tqdm.tqdm(enumerate(dataloader)):
#         x, s = batch
#         x = x.float().cuda()
#         s = s.long().cuda()
        
#         optimizer_filter_discriminator.zero_grad()

#         s_p = filter_discriminator(x.detach())
#         d_loss = F.cross_entropy(s_p, s.long())

#         d_loss.backward()
#         optimizer_filter.step()

    #for _ in range(1):
    for i_batch, batch in tqdm.tqdm(enumerate(dataloader)):
        x, s = batch
        x = x.float().cuda()
        s = s.long().cuda()

        batch_size = x.shape[0]

        # -----------------
        #  Train Filter
        # -----------------
        optimizer_filter.zero_grad()

        # sample noise as filter input
        z = torch.randn(batch_size, opt['latent_dim']).float().cuda()

        # filter a batch of images
        x_ = filter(x, z)
        s_p = filter_discriminator(x_)

        # loss measures filters's ability to fool the discriminator under constrained distortion
        ones = torch.ones(s.shape).float().cuda()
        s_c = ones-s.float()
        s_c = s_c.flatten().long()
        if not opt['use_entropy_loss']:
            f_adversary_loss = F.cross_entropy(s_p, s_c) # force it towards complement
        else:
            f_adversary_loss = -entropy_loss(s_p) # maximise entropy

        f_distortion_loss = F.l1_loss(x_, x)

        f_loss = f_adversary_loss + opt['lambd'] * torch.pow(torch.relu(f_distortion_loss-opt['eps']), 2)

        f_loss.backward()
        optimizer_filter.step()

        # ------------------------
        # Train Generator (Real/Fake)
        # ------------------------
        optimizer_generator.zero_grad()
        # sample noise as filter input
        z1 = torch.randn(batch_size, opt['latent_dim']).cuda()

        # filter a batch of images
        x_ = filter(x, z1)

        # sample noise as generator input
        z2 = torch.randn(batch_size, opt['latent_dim']).cuda()

        # sample secret
        s_ = torch.tensor(np.random.choice([0.0, 1.0], batch_size)).long().cuda()

        # generate a batch of images
        x__ = generator(x_, z2, s_)

        # loss measures generator's ability to fool the discriminator
        s_p = generator_discriminator(x__)
        g_adversary_loss = F.cross_entropy(s_p, s_)
        g_distortion_loss = F.l1_loss(x__, x)

        g_loss = g_adversary_loss + opt['lambd'] * torch.pow(torch.relu(g_distortion_loss-opt['eps']), 2)

        g_loss.backward()
        optimizer_generator.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_filter_discriminator.zero_grad()

        s_p = filter_discriminator(x_.detach())
        d_loss = F.cross_entropy(s_p, s.long())

        d_loss.backward()
        optimizer_filter.step()

        # --------------------------------
        #  Train Discriminator (Real/Fake)
        # --------------------------------
        #if i_batch % opt['discriminator_update_interval'] == 0:
        optimizer_generator_discriminator.zero_grad()

        s_p  = generator_discriminator(x)
        s_p_ = generator_discriminator(x__.detach())

        s_fake = (torch.ones(s_p_.size(0))*2).long().cuda() # create class 2
        d_rf_loss_real = F.cross_entropy(s_p, s.long())
        d_rf_loss_fake = F.cross_entropy(s_p_, s_fake)

        d_rf_loss = (d_rf_loss_real + d_rf_loss_fake) / 2

        d_rf_loss.backward()
        optimizer_generator_discriminator.step()

    print('loss/d_loss', d_loss.item(), i_batch + i_epoch*len(dataloader))
    print('loss/f_loss', f_loss.item(), i_batch + i_epoch*len(dataloader))
    print('loss/d_rf_loss', d_rf_loss.item(), i_batch + i_epoch*len(dataloader))
    print('loss/g_loss', g_loss.item(), i_batch + i_epoch*len(dataloader))

class IllustrativeDataset(torch.utils.data.Dataset):
    def __init__(self, means, stds, nb_samples_per_class):
        self.nb_classes = len(means)
        x = []
        s = []
        for c in range(self.nb_classes):
            x.append(np.random.normal(loc=means[c], scale=stds[c], size=(nb_samples_per_class, len(means[c]))))
            s.append(np.repeat(c, nb_samples_per_class))
        self.x = np.concatenate(x)
        self.s = np.concatenate(s)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.s[idx]

class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        self.filter = nn.Sequential(
            nn.Linear(2*2, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        
    def forward(self, x, z):
        return self.filter(torch.cat((x, z), dim=1))

class Generator(nn.Module):
    def __init__(self, nb_conditional_classes):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(2*3, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        self.embedding = nn.Embedding(nb_conditional_classes, 2)
    def forward(self, x, z, s):
        c = self.embedding(s)
        return self.generator(torch.cat((x, z, c), dim=1))
    
class Discriminator(nn.Module):
    def __init__(self, nb_classes):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, nb_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

def evaluate_privacy_and_utility(filter, generator, train_loader, valid_loader):
    # create adversarys
    adversary_f = Discriminator(2).cuda()
    adversary_fg = Discriminator(2).cuda()
    
    optimizer_adversary_f = torch.optim.Adam(params=adversary_f.parameters(), lr=1e-3)
    optimizer_adversary_fg = torch.optim.Adam(params=adversary_fg.parameters(), lr=1e-3)
    
    adversary_f.train()
    adversary_fg.train()
    
    for _ in range(5):
        for i_batch, batch in tqdm.tqdm(enumerate(train_loader)):
            x, s = batch
            x = x.float().cuda()
            s = s.long().cuda()

            batch_size = x.shape[0]
            
            # ----------------------------------
            # Train filter adversary
            # ----------------------------------
            optimizer_adversary_f.zero_grad()
            # sample noise as filter input
            z1 = torch.randn(batch_size, 2).float().cuda()

            # filter a batch of images
            x_ = filter(x, z1)
            s_f = adversary_f(x_.detach())
            f_adversary_loss = F.cross_entropy(s_f, s)

            f_adversary_loss.backward()
            optimizer_adversary_f.step()

            # ----------------------------------
            # Train filter + generator adversary
            # ----------------------------------
            optimizer_adversary_fg.zero_grad()
            
            # sample noise as filter input
            z2 = torch.randn(batch_size, 2).cuda()

            # sample secret
            s_ = torch.tensor(np.random.choice([0.0, 1.0], batch_size)).long().cuda()

            # generate a batch of images
            x__ = generator(x_.detach(), z2, s_)

            # loss measures generator's ability to fool the discriminator
            s_fg = adversary_fg(x__.detach())
            g_adversary_loss = F.cross_entropy(s_fg, s)

            g_adversary_loss.backward()
            optimizer_adversary_fg.step()
            
    adversary_f.eval()
    adversary_fg.eval()
    
    total_privacy_f = 0
    total_privacy_fg = 0
    total_utility_f = 0
    total_utility_fg = 0
    count = 0
    
    for i_batch, batch in tqdm.tqdm(enumerate(valid_loader)):
        x, s = batch
        x = x.float().cuda()
        s = s.long().cuda()

        batch_size = x.shape[0]

        z1 = torch.randn(batch_size, 2).float().cuda()
        z2 = torch.randn(batch_size, 2).float().cuda()
        s_ = torch.tensor(np.random.choice([0.0, 1.0], batch_size)).long().cuda()
        
        x_ = filter(x, z1)
        x__ = generator(x_, z2, s_)
        
        s_f = adversary_f(x_)
        s_fg = adversary_fg(x__)
        
        total_privacy_f += F.cross_entropy(s_f, s).detach().cpu().item()
        total_privacy_fg += F.cross_entropy(s_fg, s).detach().cpu().item()
        
        total_utility_f += F.l1_loss(x, x_).detach().cpu().item()
        total_utility_fg += F.l1_loss(x, x__).detach().cpu().item()
        
        count+=1
        
    return total_privacy_f/count, total_privacy_fg/count, total_utility_f/count, total_utility_fg/count
        
def main(opt):
    nb_samples_per_class = 200000
    train_dataset = IllustrativeDataset(means=[[-1, 1], [1, -1]], stds=[[0.7, 0.7], [0.7, 0.7]], nb_samples_per_class=nb_samples_per_class)
    valid_dataset = IllustrativeDataset(means=[[-1, 1], [1, -1]], stds=[[0.7, 0.7], [0.7, 0.7]], nb_samples_per_class=1000)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    filter = Filter().cuda()
    generator = Generator(nb_conditional_classes=2).cuda()
    filter_discriminator = Discriminator(2).cuda()
    generator_discriminator = Discriminator(3).cuda()

    optimizer_filter = torch.optim.Adam(params=filter.parameters(), lr=1e-3)
    optimizer_filter_discriminator = torch.optim.Adam(params=filter_discriminator.parameters(), lr=1e-3)
    optimizer_generator = torch.optim.Adam(params=generator.parameters(), lr=1e-3)
    optimizer_generator_discriminator = torch.optim.Adam(params=generator_discriminator.parameters(), lr=1e-3)
    
    nb_epochs = opt['nb_epochs']
    decision_boundaries = np.zeros((nb_epochs, 10000, 2))
    x_samples = np.zeros((nb_epochs, 10000, 2))
    for i_epoch in range(nb_epochs):
        train(i_epoch, 
          filter, optimizer_filter,
          filter_discriminator, optimizer_filter_discriminator,
          generator, optimizer_generator,
          generator_discriminator, optimizer_generator_discriminator, 
          train_loader, opt
        )
        
        # TODO: store decision boundary
        filter_discriminator.eval()
        x_sample = torch.FloatTensor(10000, 2).uniform_(-3.0, 3.0).cuda()
        decision_boundary = F.softmax(filter_discriminator(x_sample), dim=1)
        print(decision_boundary.shape)
        print(decision_boundaries.shape)
        decision_boundaries[i_epoch,:,:] = decision_boundary[:,:].detach().cpu().numpy().copy()
        x_samples[i_epoch,:,:] = x_sample[:,:].cpu().numpy().copy()
        
    filter.eval()
    generator.eval()
    filter_discriminator.eval()
    generator_discriminator.eval()
    return filter, generator, filter_discriminator, generator_discriminator, train_dataset, decision_boundaries, x_samples

def scatterplot(data, error, label, point_labels, ax, color):
    ax.errorbar([x[0] for x in data], [x[1] for x in data], error, linestyle='-', label=label,
            color=color)
    for i, txt in enumerate(point_labels):
        xy = (data[i][0]-0.02, data[i][1]+0.08)
        ax.annotate(r'$\epsilon = {}$'.format(txt), xy, fontsize=8)

dataset = IllustrativeDataset(means=[[-1, 1], [1, -1]], stds=[[0.7, 0.7], [0.7, 0.7]], nb_samples_per_class=1000)
fig = plt.figure(figsize=(10, 10))
sns.scatterplot(dataset.x[:,0], dataset.x[:,1], style=dataset.s)
plt.show()

opts = []
epss = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
for use_entropy_loss in [False, True]:
    for eps in epss:
        opt = {
            'latent_dim' : 2,
            'use_entropy_loss' : use_entropy_loss,
            'lambd' : 100000,
            'eps' : eps,
            'nb_epochs' : 5,
            'discriminator_update_interval' : 1
        }
        opts.append(opt)

nb_runs = 5
entropy_runs = []
likelihood_runs = []
for run in range(nb_runs):
    print("run: ", run)
    results = []
    for opt in opts:
        filter, generator, filter_discriminator, generator_discriminator, train_dataset, _, _ = main(opt)

        valid_dataset = IllustrativeDataset(means=[[-1, 1], [1, -1]], stds=[[0.7, 0.7], [0.7, 0.7]], nb_samples_per_class=128*10)

        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

        res = evaluate_privacy_and_utility(filter, generator, train_loader, valid_loader)
        results.append(res)
    if opt['use_entropy_loss']:
        entropy_runs.append(results)
    else:
        likelihood_runs.append(results)

sns.set_style("whitegrid")
x_f = np.zeros(len(epss))
y_f = np.zeros(len(epss))
x_fg = np.zeros(len(epss))
y_fg = np.zeros(len(epss))

x_f = np.zeros((nb_runs, len(epss)))
y_f = np.zeros((nb_runs, len(epss)))
x_fg = np.zeros((nb_runs, len(epss)))
y_fg = np.zeros((nb_runs, len(epss)))

for i, results in enumerate(runs):
    x_f[i,:] = np.array([r[0] for r in results])
    y_f[i,:] = np.array([r[2] for r in results])
    x_fg[i,:] = np.array([r[1] for r in results])
    y_fg[i,:] = np.array([r[3] for r in results])

y_f_std = np.std(y_f, axis=0)
y_fg_std = np.std(y_fg, axis=0)
    
x_f_mu = np.mean(x_f, axis=0)
y_f_mu = np.mean(y_f, axis=0)
x_fg_mu = np.mean(x_fg, axis=0)
y_fg_mu = np.mean(y_fg, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
baseline_data = np.concatenate((np.expand_dims(x_f_mu, axis=1), np.expand_dims(y_f_mu, axis=1)), axis=1)
ours_data = np.concatenate((np.expand_dims(x_fg_mu, axis=1), np.expand_dims(y_fg_mu, axis=1)), axis=1)
print(baseline_data.shape)
scatterplot(data=baseline_data, error=y_f_std, label='baseline (entropy)', point_labels=epss, ax=ax, color='blue')
scatterplot(data=ours_data, error=y_fg_std, label='ours (entropy)', point_labels=epss, ax=ax, color='green')
plt.xlabel('priavcy')
plt.ylabel('utility loss')
plt.xlim(0, 0.8)
plt.ylim(0, 1.8)
plt.legend(loc='upper left')
plt.show()
