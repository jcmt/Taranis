import numpy as np


def cclimatology(var, time, relative=False, smooth=None, window=15, israin=False):
  day = np.asarray([i.day for i in time])
  month = np.asarray([i.month for i in time])
  hour = np.asarray([i.hour for i in time])

  def get_clim(T, israin=False):
    cond = (day == T.day) & (month == T.month) & (hour == T.hour)
    if israin == True:
      return var[cond].sum(0)
    else:
      return var[cond].mean(0)

  varclim = np.zeros(var.shape,var.dtype)
  for i in range(len(time)):
    varclim[i] = get_clim(time[i])

  if smooth == 'runmean':
    varclim = runmean(varclim, window)

  if relative == True:
    varpert = (var - varclim) / varclim
  else:
    varpert = var - varclim

  return varpert, varclim


def KinKmeans(var, nk=None, tol=1e-4, n_init=100):
  ## From MJCarvalho GapStatistics
  from sklearn.cluster import KMeans

  Nd = np.size(var, axis=0)
  S = np.zeros(Nd)
  f = np.zeros(Nd)
  alpha = np.zeros(Nd)

  if not nk:
    term = 3
  else:
    term = nk

  kink = [0]
  i = 0
  while len(kink) <= term:
    ## Kmeans
    kmeans = KMeans(init='k-means++', n_clusters=i+1, n_init=n_init, tol=tol)
    T = kmeans.fit_transform(var, y=None)
    I = np.nansum(T**2,axis=0)
    S[i] = np.nansum(I,axis=0)
    ## Det. Alpha
    if i==1:
        alpha[i] = 1.0 - (3.0/(4.0*Nd))
    elif i>1:
        alpha[i] = alpha[i-1] + (1-alpha[i-1])/6.0
    ## Det. f(k)
    if i==0:
        f[i] = 1
    else:
        f[i] = S[i] / (alpha[i] * S[i-1])

    if not nk:
      index = np.r_[True, f[1:] < f[:-1]] & np.r_[f[:-1] <= f[1:], True] | \
              np.r_[True, f[1:] <= f[:-1]] & np.r_[f[:-1] < f[1:], True]
      kink = np.arange(len(f))[index]
    else:
      kink.append(0)
    i += 1

  return kink[1], f


def eof(var, neof=30, dim='field'):
  from scipy.sparse.linalg import eigs

  if var.ndim > 2:
    # (Re)Arranging Matrix ...
    data = np.empty(shape=(np.size(var, axis=0), np.size(var, axis=1) * np.size(var, axis=2)))
    for i in range(0, np.size(var, axis=0)):
      temp = var[i, 0, :]
      for j in range(1, np.size(var, axis=1)):
        temp = np.concatenate((temp, var[i, j, :]), axis=0)
      data[i, ] = temp
  else:
    data = var

  # Computing Covariance Matrix
  c = np.cov(data, rowvar=0)

  # Computing EOFs ...
  s, eof_data = eigs(c, neof, which='LR')
  s = s.real
  eof_data = eof_data.real
  pc = np.dot(data, eof_data)
  pcvar = s / np.sum(s) * 100

  if dim == 'field':
    eof_temp = np.empty(shape=[neof, np.size(var, axis=1), np.size(var, axis=2)])
    for i in range(0, neof):
      aux = np.reshape(eof_data[:, i], [np.size(var, axis=1), np.size(var, axis=2)])
      eof_temp[i, :, :] = aux
    eof_data = np.copy(eof_temp)
  return eof_data, pc, pcvar

def svd(var1, var2, neof):

  if (len(var1.shape) > 1 and len(var2.shape) > 1):
    # (Re)Arranging Matrix ...
    data1 = np.empty(shape=(np.size(var1, axis=0), np.size(var1, axis=1) * np.size(var1, axis=2)))
    data2 = np.empty(shape=(np.size(var2, axis=0), np.size(var2, axis=1) * np.size(var2, axis=2)))
    for i in range(0, np.size(var1, axis=0)):
      temp_dust = var1[i, 0, :]
      temp_msl = var2[i, 0, :]
      for j in range(1, np.size(var1, axis=1)):
        temp_dust = np.concatenate((temp_dust, var1[i, j, :]), axis=0)
        temp_msl = np.concatenate((temp_msl, var2[i, j, :]), axis=0)
      data1[i, ] = temp_dust
      data2[i, ] = temp_msl
  else:
    data1 = var1
    data2 = var2

  # Computing Covariance Matrix ...
  c = np.dot(data1.T, data2) / np.size(data1, axis=0)
  u, s, v = np.linalg.svd(c, full_matrices=True)

  SVD1 = u
  SVD2 = v.T
  a = np.dot(data1, u)
  b = np.dot(data2, v)
  l = np.dot(a.T, b)
  sfc = np.diag(s)**2 / np.sum(np.diag(s)**2)

  svd1_temp = np.empty(shape=[neof, np.size(var1, axis=1), np.size(var1, axis=2)])
  svd2_temp = np.empty(shape=[neof, np.size(var2, axis=1), np.size(var2, axis=2)])
  for i in range(0, neof):
    aux1 = np.reshape(SVD1[:, i], [np.size(var1, axis=1), np.size(var1, axis=2)])
    svd1_temp[i, :, :] = aux1

    aux2 = np.reshape(SVD2[:, i], [np.size(var2, axis=1), np.size(var2, axis=2)])
    svd2_temp[i, :, :] = aux2

  svd1 = svd1_temp
  svd2 = svd2_temp

  return svd1, svd2, sfc, l, a, b

def runmean(x, N):
  import scipy.ndimage

  y = scipy.ndimage.filters.convolve1d(np.tile(x, [3]+[1]*(x.ndim-1) ), \
                                       np.ones(N)/float(N), axis=0, mode='nearest')
  return y[x.shape[0]:-x.shape[0]]
