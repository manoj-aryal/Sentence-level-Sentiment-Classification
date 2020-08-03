import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sentence_classification import SST
from network import Network

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=32,
	help='Batch size for mimi-batch training and evaluating. Default: 32')
parser.add_argument('--epoch', type=int, default=50,
	help='Number of training epoch. Default: 50')
parser.add_argument('--mode', type=str, default='GRU',
	help='Three modes for the first three tasks: \'GRU\', \'LSTM\', \'Attention\'')

args = parser.parse_args()

def train(dataloader, model, optimizer):
	dataloader.restart('train', args.batchsize)

	total_loss = []
	total_precise = []
	while True:
		optimizer.zero_grad()

		batch = dataloader.get_next_batch('train')
		if batch is None:
			break
		sent = torch.from_numpy(batch['sent']).long().cuda()
		sent_length = torch.from_numpy(batch['sent_length']).long().cuda()
		label = torch.from_numpy(batch['label']).long().cuda()

		logit, loss = model(sent, sent_length, label)

		loss.backward()
		optimizer.step()

		total_precise.append((torch.max(logit, dim=1)[1] == label).float().mean().cpu().data.numpy())
		total_loss.append(loss.cpu().data.numpy())

	print('[train]loss: %f, precise: %f' % (np.mean(total_loss), np.mean(total_precise)))

def eval(dataloader, model):
	'''
	evaluate on dev set
	'''
	dataloader.restart('dev', args.batchsize, shuffle=False)

	total_precise = []
	while True:

		batch = dataloader.get_next_batch('dev')
		if batch is None:
			break
		sent = torch.from_numpy(batch['sent']).long().cuda()
		sent_length = torch.from_numpy(batch['sent_length']).long().cuda()
		label = torch.from_numpy(batch['label']).long().cuda()

		logit = model(sent, sent_length)

		total_precise.append((torch.max(logit, dim=1)[1] == label).float().mean().cpu().data.numpy())

	precise = np.mean(total_precise)
	print('[dev]precise: %f' % (precise))
	return precise

def predict(dataloader, model):
	'''
	predict labels for test set 
	'''
	dataloader.restart('test', args.batchsize, shuffle=False)

	with open('prediction', 'w') as w:

		while True:

			batch = dataloader.get_next_batch('test')
			if batch is None:
				break
			sent = torch.from_numpy(batch['sent']).long().cuda()
			sent_length = torch.from_numpy(batch['sent_length']).long().cuda()

			logit = model(sent, sent_length)
			prediction = torch.max(logit, dim=1)[1]

			for x in prediction:
				w.write('%d\n' % (x))

def main(dataloader, model, optimizer):
	best_precise = {'dev':0., 'epoch':-1}
	for e in range(args.epoch):
		print('*' * 30)
		print('trainging epoch %d...' % (e))
		train(dataloader, model, optimizer)
		dev_precise = eval(dataloader, model)
		if dev_precise > best_precise['dev']:
			best_precise['dev'] = dev_precise
			best_precise['epoch'] = e
			print('make prediction on test set...')
			predict(dataloader, model)
		print('[best dev performance]epoch: %d, dev: %f' \
					% (best_precise['epoch'], best_precise['dev']))

if __name__ == '__main__':
	print('load data...')
	dataloader = SST()
	print('create model...')
	if args.mode not in ['GRU', 'LSTM', 'Attention']:
		raise ValueError('Please use `--mode [GRU/LSTM/Attention]`')
	model = Network(dataloader.emb, mode=args.mode)
	model.cuda()
	print('create optimizer...')
	optimizer = optim.Adam(model.parameters())
	print('training')
	main(dataloader, model, optimizer)
