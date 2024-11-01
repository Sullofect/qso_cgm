import astroplan
from astroplan import Observer
from astropy.time import Time

# Define OBS
las = Observer.at_site("Las Campanas Observatory", timezone="UTC")


# Calcualte Parallactic angle
time
para = las.parallactic_angle(time, target)