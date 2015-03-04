"""Taranis ...
"""

classifiers = """\
Development Status :: alpha
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: European Union Public Licence - EUPL v.1.1
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
"""

from numpy.distutils.core import Extension
from numpy.distutils.command.install import install
from glob import glob

class my_install(install):
    def run(self):
        install.run(self)

        print '''
        enjoy taranis
        '''


doclines = __doc__.split("\n")

if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(name = "taranis",
          version = 0,
          description = doclines[0],
          long_description = "\n".join(doclines[2:]),
          author = "Joao Teixeira",
          author_email = "jcmt87@gmail.com",
          url = "NA",
          packages = ['taranis'],
          license = 'EUPL',
          platforms = ["any"],
          ext_modules = [],
          data_files=[('taranis/cmap', glob('taranis/cmap/*')),
                      ('taranis/colormaps', glob('taranis/colormaps/*')),
                      ('taranis/', glob('taranis/*.so'))],
          classifiers = filter(None, classifiers.split("\n")),
          cmdclass={'install': my_install},
          )

