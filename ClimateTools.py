import numpy as np
from datetime import date

def cclimatology(var, time, calendar='gregorian', dt=None):
    '''
    Calculates the daily Perturbation from the Climatology and the Climatology
    usage:
       varpert, carclim = cclimatology(var, time, calendar)

    where:
       var is a 3D variable [time, lat, lon]. Must have at least daily data
       time is a datetime with whit the same elements as var[time]
    '''

    start = time[0].year
    end = time[-1].year
    if not dt:
        dt = (time[1] - time[0]).total_seconds() / 86400.0    # dt in days

    year_list = np.arange(start, end+1)

    for i in year_list:
        if calendar == 'gregorian':
            delta = date(i+1, 1, 1) - date(i, 1, 1)
            delta = delta.days
        elif (calendar == '365_day') | (calendar == 'no_leap') | (calendar == 'noleap'):
            delta = 365
        elif calendar == '360':
            delta = 360
        else:
            delta = date(i+1, 1, 1) - date(i, 1, 1)
            delta = delta.days

        if i == year_list[0]:
            day_list = np.arange(0, delta/dt, 1, dtype=np.int)
        else:
            day_list = np.hstack((day_list, np.arange(0, delta/dt, 1, dtype=np.int)))

    # day_list += 1    # just to be human readable (useful for debugging)

    varpert = np.empty(np.shape(var))
    varclim = np.empty(shape=[len(day_list), np.size(var, axis=1), np.size(var, axis=2)])
    for i in range(day_list.max()):
        index = day_list == i
        varclim[i, :, :] = np.mean(var[index, :, :], axis=0)
        varpert[index, :, :] = var[index, :, :] - varclim[i, :, :]

    return varpert, varclim


def eof(var, neof, dim='field'):
    from scipy.sparse.linalg import eigs

    if len(var.shape) > 1:
        # (Re)Arranging Matrix ...
        data = np.empty(shape=(np.size(var, axis=0), np.size(var, axis=1) * np.size(var, axis=2)))
        for i in range(0, np.size(var, axis=0)):
            temp = var[i, 0, :]
            for j in range(1, np.size(var, axis=1)):
                temp = np.concatenate((temp, var[i, j, :]), axis=0)
            data[i, ] = temp
    else:
        data = var
    del temp

    # Computing Covariance Matrix
    c = np.cov(data, rowvar=0)

    # Computing EOFs ...
    s, eof_data = eigs(c, k=neof, which='LR')
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
