import os
import sys
import time
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from math import sqrt
import torch.optim.lr_scheduler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from random import random
from sklearn import preprocessing
from torch.autograd import Variable
from utils import Logger
from args import LOGGER_DISPLAY, LOGGER_DISPLAY_VAR
log = Logger(display=LOGGER_DISPLAY, ori_var=LOGGER_DISPLAY_VAR)