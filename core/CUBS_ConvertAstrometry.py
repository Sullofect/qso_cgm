import os
import glob
import numpy as np
from astropy.io import fits

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

def save_all_headers():
    path = '../../MUSEQuBES+CUBS/CUBS_hdr/ESO/'
    outdir = os.path.join(path, 'header_only')
    os.makedirs(outdir, exist_ok=True)

    files = glob.glob(os.path.join(path, '*.fits'))

    for file in files:
        print('Processing:', file)
        basename = os.path.basename(file)
        root = os.path.splitext(basename)[0]

        cubename = root.split('_')[0]
        after_cubename = root[len(cubename):]

        # Default: assume filename already uses canonical name
        cubename_canonical = cubename

        # Case 1: cubename is already a canonical name (dictionary key)
        if cubename in object_aliases:
            cubename_canonical = cubename

        # Case 2: cubename is an alias -> find which canonical name it maps to
        else:
            for canonical_name, aliases in object_aliases.items():
                if cubename in aliases:
                    cubename_canonical = canonical_name
                    break

        if 'gaia' in after_cubename.lower():
            after_cubename = '_gaia'
        else:
            after_cubename = ''

        # Read header from extension 1
        with fits.open(file) as hdul:
            header1 = hdul[1].header.copy()

        # Print original NAXIS1 / NAXIS2
        naxis1_1 = header1.get('NAXIS1', 'NOT FOUND')
        naxis2_1 = header1.get('NAXIS2', 'NOT FOUND')
        header1['ORIGNAX1'] = header1.get('NAXIS1', -1)
        header1['ORIGNAX2'] = header1.get('NAXIS2', -1)

        # Save as header-only FITS
        outfile = os.path.join(outdir, f'{cubename_canonical}{after_cubename}_header.fits')
        hdu = fits.PrimaryHDU(header=header1)
        hdu.writeto(outfile, overwrite=True)

        print(f'Input file: {basename}')
        print(f'Saved: {os.path.basename(outfile)}')
        print(f'  Original header: NAXIS1={naxis1_1}, NAXIS2={naxis2_1}')

save_all_headers()


# Check the saved header-only files
def check_headers():
    path = '../../MUSEQuBES+CUBS/CUBS_hdr/ESO/header_only/'
    for cubename in object_aliases.keys():
        for suffix in ['', '_gaia']:
            filename = f'{cubename}{suffix}_header.fits'
            filepath = os.path.join(path, filename)
            if os.path.exists(filepath):
                with fits.open(filepath) as hdul:
                    header = hdul[0].header
                    naxis1 = header.get('ORIGNAX1', 'NOT FOUND')
                    naxis2 = header.get('ORIGNAX2', 'NOT FOUND')
                    print(f'{filename}: NAXIS1={naxis1}, NAXIS2={naxis2}')
            else:
                print(f'File not found: {filename}')
check_headers()






