

def get_obs(FILE, start=False, stop=False):
  from datetime import datetime
  import numpy as np


  dtype = [int, int, int, int, int, float, float, float, float, float, float, float]
  known_st = []
  raw = np.genfromtxt(FILE, delimiter='\t', skip_header=1, dtype=dtype)

  obs = {}
  obs['time'] = np.asarray([datetime(raw[i][1], raw[i][2], raw[i][3], raw[i][4]) for i in range(len(raw))])

  if (start!=False) & (stop!=False):
    ini  = np.where(obs['time'] == start)[0][0]
    fini = np.where(obs['time'] == stop)[0][0]
    obs['time'] = obs['time'][ini:fini]
  elif (start!=False) & (stop==False):
    ini  = np.where(obs['time'] == start)[0][0]
    fini = len(raw)
    obs['time'] = obs['time'][ini:fini]
  elif (start==False) & (stop!=False):
    ini  = 0
    fini = np.where(obs['time'] == stop)[0][0]
    obs['time'] = obs['time'][ini:fini]
  else:
    ini  = 0
    fini = len(raw)

  VARS = [['slp', 'mslp', 'tmp', 'rh', 'wdir', 'wmag', 'rain'],\
          [ 5,     6,      7,      8,    9,      10,     11]]

  for j in range(len(VARS[0])):
    obs[VARS[0][j]]  = np.asarray([raw[i][VARS[1][j]] for i in range(ini, fini)])
    obs[VARS[0][j]][obs[VARS[0][j]] == -990] = np.nan

  obs.update(isknown(raw[0][0]))
  return obs

def isknown(stcode):
  knownst = {}
  knownst['name'] = ['Cabo Carvoeiro', 'Sagres', 'Geofisico', 'Sines',  'Porto',  'Coimbra', 'Faro',   'Evora', 'Viseu',   'Beja',   'Vila Real', 'Penhas Douradas', 'Castelo Branco', 'Portalegre', 'Braganca', 'Gago Coutinho']
  knownst['code'] = [1200531,          1200533,  1200535,     1200541,  1200545,  1200548,   1200554,  1200558, 1200560,   1200562,  1200567,     1200568,           1200570,          1200571,      1200575,    1200579]
  knownst['lat']  = [39.3614,          37.0128,  38.7190,     37.9545,  41.2335,  40.1576,   37.0165,  38.5365, 40.7149,   38.0257,  41.2742,     40.4113,           39.8394,          39.2941,      41.8038,    38.7662]
  knownst['lon']  = [-9.4069,          -8.9490,  -9.1497,     -8.8382,  -8.6813,  -8.4685,   -7.9719,  -7.8879, -7.8959,   -7.8673,  -7.7171,     -7.5586,           -7.4786,          -7.4213,      -6.7428,    -9.1274]

  try:
    idd = knownst['code'].index(stcode)
    return {i:knownst[i][idd] for i in knownst.keys()}
  except ValueError:
    print('Station not found...')
