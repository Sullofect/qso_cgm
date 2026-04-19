import numpy as np


object_aliases = {"J0110-1648": ["Q0110-1648"],
                  "J0454-6116": ["Q0454-6116"],
                  "J2135-5316": ["Q2135-5316"],
                  "J0119-2010": ["Q0119-2010"],
                  "HE0246-4101": ["Q0248-4048", "J0248-4048"],
                  "J0028-3305": ["Q0028-3305"],
                  "HE0419-5657": ["Q0420-5650", "J0420-5650"],
                  "PKS2242-498": ["Q2245-4931", "J2245-4931"],
                  "PKS0355-483": ["Q0357-4812", "J0357-4812"],
                  "HE0112-4145": ["Q0114-4129", "J0114-4129"],
                  "HE2305-5315": ["Q2308-5258", "J2308-5258"],
                  "HE0331-4112": ["Q0333-4102", "J0333-4102"],
                  "J0111-0316": ["Q0111-0316"],
                  "J0154-0712": ["Q0154-0712"],
                  "HE2336-5540": ["Q2339-5523", "J2339-5523"],
}






def switch_astrometry(filename_input=None, filename_output=None):
    hdul_input = fits.open(filename_input)
    header_input = hdul_input[1].header
    data_input = hdul_input[1].data

    # Find the correct Gaia-corrected header
    cubename = filename_input.split('/')[-1].split('_')[0]
    cubename_canonical = cubename
    if cubename in object_aliases:
        cubename_canonical = cubename
    else:
        for canonical_name, aliases in object_aliases.items():
            if cubename in aliases:
                cubename_canonical = canonical_name
                break

    #







