import netCDF4 as n4
import numpy as np


class get:
  def __init__(self, path):
    self.path = path
    self.nc = n4.Dataset(path)

  def close(self):
    self.nc.close()

  def variables(self):
    varname = np.asarray([v for v in self.nc.variables])
    return varname

  def dim(self):
    nt = len(self.nc.dimensions['time'])
    nx = len(self.nc.dimensions['x'])
    ny = len(self.nc.dimensions['y'])
    nz = len(self.nc.dimensions['z'])

    return np.array([nt, nz, ny, nx])

  def lat(self):
    '''
    Returns the latitude array
    '''

    LAT = self.nc.variables['lat'][...]
    return LAT

  def lon(self):
    '''
    Returns the longitude array
    '''

    LON = self.nc.variables['lon'][...]
    return LON

  def height(self, tstep=None, nlev=':', ny=':', nx=':'):
    '''
    Returns the height of the model levels at a given time

    usage:
        height(tstep)

        tstep is the time instant, if not specified all the written times
        will be used
    '''

    if tstep==None:
      Z = self.getvar('ZH', tstep=':', nlev=nlev, ny=ny, nx=nx)

    else:
      Z = self.getvar('ZH', tstep=tstep, nlev=nlev, ny=ny, nx=nx)
    return Z

  def time(self, tstep=None):
    '''
    Returns a datetime
    '''

    if tstep==None:
      t = self.nc.variables['time'][...]
    else:
      t = self.nc.variables['time'][tstep]

    cal = self.nc.variables['time'].calendar
    unit = self.nc.variables['time'].units
    TIME = n4.num2date(t, units=unit, calendar=cal)
    return TIME

  def getvar(self, var, tstep=':', nlev=':', ny=':', nx=':'):
    '''
    Returns the data from a given variable in the wrfout file

    usage:
       getvar(var, tstep)

       var is a string with the variable name (example 'U10')
       tstep is the time instant, if not specified all the written times will
       be used

    Warning:
       For the variable P, PH and T, their base state will be added
    '''

    if len(self.nc.variables[var].dimensions) == 4:
      SLICE = str(tstep) + ',' + str(nlev) + ',' + str(ny) + ',' + str(nx)
    elif self.nc.variables[var].dimensions == 3:
      SLICE = str(tstep) + ',' + str(ny) + ',' + str(nx)
    elif self.nc.variables[var].dimensions == 2:
      SLICE = str(ny) + ',' + str(nx)

    VAR = eval("self.nc.variables['" + var + "'][" + SLICE + "]")

    if (var == 'P') | (var == 'PH'):
      VAR += eval("self.nc.variables['" + var + "B'][" + SLICE + "]")
    if (var == 'T'):
      VAR += 300

    return VAR


  def pcolor(self, VAR, tstep=None, colormap=None, colorbar=True, level=0, shading='flat', norm=None):
    '''
    lat-lon plot on a base map

    usage:
       pcolor(VAR, colormap, colorbar, tstep, level, shading, norm)

       VAR is a wrfout variable (string) or a 2D numpy array
       if VAR is tring a tstep and level must be given to acquire the
       variable. IF NOT the first level and time will be used
       shading can be one of: flat (default), interp (contourf) or None
       (pcolor)
    '''
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import ticks

    if tstep==None:
      return "A time step must be specified..."
    else:
      if isinstance(VAR, str):
        if len(self.nc.variables[VAR].dimensions) == 4:
          VAR = self.getvar(VAR, tstep=tstep, nlev=level, ny=':', nx=':')
        elif len(self.nc.variables[VAR].dimensions) == 3:
          VAR = self.getvar(VAR, tstep=tstep)
        elif len(self.nc.variables[VAR].dimensions) == 2:
          VAR = self.getvar(VAR)

      proj = 'lcc'

      lon_0 = self.nc.cenlon
      lat_0 = self.nc.cenlat
      llcrnrlat = self.lat()[4, 4]
      urcrnrlat = self.lat()[-4,-4]
      llcrnrlon = self.lon()[4, 4]
      urcrnrlon = self.lon()[-4, -4]

      res = 'i'
      if self.nc.dx < 15:
        res = 'h'

      plt.figure()
      m = Basemap(projection=proj, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, \
                    llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, \
                    lat_0=lat_0, lon_0=lon_0, resolution=res)

      m.drawcoastlines(color='black', linewidth=2)
      m.drawcountries(linewidth=1.5)

      parallels = ticks.loose_label(self.lat().min(),self.lat().max())
      m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=14)
      meridians = ticks.loose_label(self.lon().min(),self.lon().max())
      m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=14)

      x, y = m(self.lon(), self.lat())

      if shading == 'interp':
        levels = np.linspace(VAR.min(), VAR.max(), 200)
        cs = plt.contourf(x, y, VAR, cmap=colormap, levels=levels)
      else:
        cs = m.pcolormesh(x, y, VAR, cmap=colormap, shading=shading)

      plt.title(self.time(tstep=tstep))

      if colorbar == True:
        fmt = plt.matplotlib.ticker.FormatStrFormatter("%.1f")
        clev = np.arange(VAR.min(), VAR.max(), (VAR.max() - VAR.min())/ 8.)
        cbar = m.colorbar(cs, location='right', ticks=clev, format=fmt, pad='10%')
      return self.time(tstep=tstep)

  def CrossPcolor(self, VAR, tstep=1, latitude=None, longitude=None, colormap=None, \
                  colorbar=False, norm=None, ymax=10000, ymin=0, shading='flat', lev=None):
    import matplotlib.pyplot as plt
    plt.figure()

    if latitude == None and longitude == None:
      return('A latitude and longitude range must be chosen...')

    elif latitude == None:
      pos_lon = np.argmin(abs(self.lon()[1, :] - longitude))
      pos_lat = slice(0, np.size(self.lat(), axis=0))
      y = self.height(tstep=tstep, nlev=':', ny=pos_lat, nx=pos_lon)
      x = np.tile(self.lat()[:, pos_lon], (self.dim()[1], 1))
      xlabel = 'Longitude ($^\circ$)'

    elif longitude == None:
      pos_lon = slice(0, np.size(self.lon(), axis=1))
      pos_lat = np.argmin(abs(self.lat()[:, 1] - latitude))
      y = self.height(tstep=tstep, ny=pos_lat, nx=pos_lon)
      x = np.tile(self.lon()[pos_lat, :], (self.dim()[1], 1))
      xlabel = 'Latitude ($^\circ$)'

    else:
      return('I cant deal with this.. yet!!!')

    if isinstance(VAR, str):
      if len(self.nc.variables[VAR].dimensions) == 4:
        VAR = self.getvar(VAR, tstep=tstep, nlev=':', ny=pos_lat, nx=pos_lon)
      elif len(self.nc.variables[VAR].dimensions) == 3:
        VAR = self.getvar(VAR, tstep=tstep, ny=pos_lat, nx=pos_lon)
      elif len(self.nc.variables[VAR].dimensions) == 2:
        VAR = self.getvar(VAR, ny=pos_lat, nx=pos_lon)

    else:
      VAR = np.squeeze(VAR[:, pos_lat, pos_lon])

    if shading == 'interp':
      if lev == None:
        levels = np.linspace(VAR.min(), VAR.max(), 200)
      else:
        levels = np.linspace(lev[0], lev[1], 100)

      cs = plt.contourf(x[0:VAR.shape[0], :], y[0:VAR.shape[0], :], VAR, cmap=colormap, norm=norm, levels=levels)
    else:
      cs = plt.pcolormesh(x[0:VAR.shape[0], :], y[0:VAR.shape[0], :], VAR, cmap=colormap, norm=norm, shading=shading)

    if colorbar == True:
      fmt = plt.matplotlib.ticker.FormatStrFormatter("%.1f")
      clev = np.arange(VAR.min(), VAR.max(), (VAR.max() - VAR.min())/ 8.)
      cbar = plt.colorbar(cs, ticks=clev, format=fmt, norm=norm)

    plt.title(self.time()[tstep])
    plt.xlabel(xlabel)
    plt.ylabel('Height (m)')
    plt.ylim([ymin, ymax])
    plt.xlim(x.min(), x.max())


def myround(x, base=5):
  x *= 100
  y = int(base * round(float(x)/base))
  y /= 100.0
  return y


def interp3d(A, PR, val):
  s = np.shape(PR)  #size of the input arrays
  ss = [s[1], s[2]] # shape of 2d arrays
  interpVal = np.empty(ss, np.float32)
  ratio = np.zeros(ss, np.float32)

  #  the LEVEL value is determine the lowest level where P<=val
  LEVEL = np.empty(ss, np.int32)
  LEVEL[:, :] = -1 #value where PR<=val has not been found
  for K in range(s[0]):
    #LEVNEED is true if this is first time PR<val.
    LEVNEED = np.logical_and(np.less(LEVEL, 0), np.less(PR[K, :, :], val))
    LEVEL[LEVNEED] = K
    ratio[LEVNEED] = (val-PR[K, LEVNEED]) / (PR[K-1, LEVNEED] - PR[K, LEVNEED])
    interpVal[LEVNEED] = ratio[LEVNEED] * A[K, LEVNEED] + (1-ratio[LEVNEED]) * A[K-1, LEVNEED]
    LEVNEED = np.greater(LEVEL, 0)
  # Set unspecified values to value of A at top of data:
  LEVNEED = np.less(LEVEL, 0)
  interpVal[LEVNEED] = A[s[0]-1, LEVNEED]
  return interpVal

