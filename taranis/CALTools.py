import numpy as np
import h5py as h5
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap


class getCALIPSO:
  def __init__(self, path):
    self.path = path
    self.f = h5.File(path, 'r')

  def close(self):
    self.f.close()

  def lat(self):
    '''
    Returns satellite swath latitude
    '''

    LAT = self.f['Latitude'][:, 0]
    return LAT

  def lon(self):
    '''
    Returns satellite swath latitude
    '''

    LON = self.f['Longitude'][:, 0]
    return LON

  def height(self):
    '''
    Returns Lidar Data Altitudes in meters
    '''

    Z = self.f['metadata']['Lidar_Data_Altitudes'][0, :]*1000
    return Z

  def time(self):
    '''
    Returns a datetime
    '''

    from netCDF4 import num2date

    TIME = num2date(self.f['Profile_Time'][:, 0],
                    units='seconds since 1993-01-01 00:00:00', calendar='gregorian')
    return TIME

  def getvar(self, var):
    '''
    Returns data from the HDF5 Dataset
    '''

    VAR = self.f[var][...].T
    if var == 'Surface_Elevation':
      VAR *= 1000
    return VAR

  def crop(self, minlat, maxlat, minlon, maxlon):
    '''
    Finds the index for a given region (to crop datasets)

    usage:
        index = crop(minlat, maxlat, minlon, maxlon)

        minlat the latitude minimum
        maxlat the latitude maximum
        minlon the longitude minimum
        maxlon the longitude maximum

    returns:
        index a vector containing the elements of latitude and longitude in
        which the given region is contained
    '''

    index = np.where((self.lat() < maxlat) & (self.lat() > minlat) &
                     (self.lon() > minlon) & (self.lon() < maxlon))[0]
    return index

  def swath(self):
    '''
    Plots the satellite swath
    '''

    m = Basemap(projection='mill', resolution='i', \
                llcrnrlat=15, urcrnrlat=60,\
                llcrnrlon=-30, urcrnrlon=25)
    m.drawcoastlines(color='black', linewidth=2)
    m.drawcountries(linewidth=1.5, color=[0.5, 0.5, 0.5])

    parallels = np.arange(-90., 90, 20)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=16, fontweight='bold')
    meridians = np.arange(0., 360., 20)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=16, fontweight='bold')

    x, y = m(self.lon(), self.lat())
    cs = m.scatter(x, y, color='red')

  def plot(self, x, y, var):
    '''
    Plots a given cross-section of the satellite swath

    usage:
        plot(x, y, var)

        x is the data for the x-axis (can be 1D or 2D)
        y is the data for the y-axis (same dimentions as x)
        var is the data to plot (2D)
    '''

    import matplotlib.pyplot as plt

    if (np.size(x.shape) < 2) and (np.size(y.shape) < 2):
      x, y = np.meshgrid(x, y)
    elif (np.size(x.shape) == 2) and (np.size(y.shape) == 2):
      pass
    else:
      return ('I can not deal with x and y... Problem with dimentions???')

    levels = np.linspace(0, 0.1, 24)

    colormap, norm, ticks = loadcolormap('calipso-backscatter.cmap', 'calipso')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = plt.pcolormesh(x, y, var, cmap=colormap, norm=norm)

    plt.ylim([0, y.max()])
    plt.xlim([x.min(), x.max()])

    ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = mpl.colorbar.ColorbarBase(ax2, cmap=colormap,  ticks=ticks, \
                                     norm=norm, orientation="vertical", extend='both', format=SciFormatter())
    return ax


def loadcolormap(filename, name):
  """"Returns a tuple of matplotlib colormap, matplotlib norm,
  and a list of ticks loaded from the file filename in format:

  BOUNDS
  from1 to1 step1
  from2 to2 step2
  ...

  TICKS
  from1 to1 step1
  from2 to2 step2

  COLORS
  r1 g1 b1
  r2 g2 b2
  ...

  UNDER_OVER_BAD_COLORS
  ro go bo
  ru gu bu
  rb gb bb

  Where fromn, ton, stepn are floating point numbers as would be supplied
  to numpy.arange, and rn, gn, bn are the color components the n-th color
  stripe. Components are expected to be in base10 format (0-255).
  UNDER_OVER_BAD_COLORS section specifies colors to be used for
  over, under and bad (masked) values in that order.

  Arguments:
      filename    -- name of the colormap file
      name        -- name for the matplotlib colormap object

  Returns:
      A tuple of: instance of ListedColormap, instance of BoundaryNorm, ticks.

  FROM ccplot
  """
  import os
  import taranis

  bounds = []
  ticks = []
  rgbarray = []
  specials = []
  mode = "COLORS"

  path = os.path.join(os.path.join(os.path.dirname(__file__)), 'cmap')
  fp = None
  fp = open(os.path.join(path, filename), "r")

  if fp == None: return("%s: colormap File not found" % filename)

  try:
    lines = fp.readlines()
    for n, s in enumerate(lines):
      s = s.strip()
      if len(s) == 0: continue
      if s in ("BOUNDS", "TICKS", "COLORS", "UNDER_OVER_BAD_COLORS"):
        mode = s
        continue

      a = s.split()
      if len(a) not in (3, 4):
        raise ValueError("Invalid number of fields")

      if mode == "BOUNDS":
        bounds += list(np.arange(float(a[0]), float(a[1]), float(a[2])))
      elif mode == "TICKS":
        ticks += list(np.arange(float(a[0]), float(a[1]), float(a[2])))
      elif mode == "COLORS":
        rgba = [int(c)/256.0 for c in a]
        if len(rgba) == 3: rgba.append(1)
        rgbarray.append(rgba)
      elif mode == "UNDER_OVER_BAD_COLORS":
        rgba = [int(c)/256.0 for c in a]
        if len(rgba) == 3: rgba.append(1)
        specials.append(rgba)

  except IOError, err:
    return(err)
  except ValueError, err:
    return("Error reading `%s' on line %d: %s" % (filename, n+1, err))

  if (len(rgbarray) > 0):
    colormap = mpl.colors.ListedColormap(rgbarray, name)
    try:
      colormap.set_under(specials[0][:3], specials[0][3])
      colormap.set_over(specials[1][:3], specials[1][3])
      colormap.set_bad(specials[2][:3], specials[2][3])
    except IndexError: pass
  else:
    colormap = None

  if len(bounds) == 0:
    norm = None
  else:
    norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
  if len(ticks) == 0: ticks = None
  return (colormap, norm, ticks)

class SciFormatter(mpl.ticker.Formatter):
  def __call__(self, x, pos=None):
    import math

    if x == 0.0:
      return "0.0"
    y = math.log(abs(x), 10)
    n = int(math.floor(y))
    if n < -1 or n > 2:
      return "%.1fx10$^{%d}$" % (x/10**n, n)
    else:
      return "%.1f" % (x,)
