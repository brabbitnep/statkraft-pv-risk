import pvlib
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
print("Canadian Solar modules found:")
for mod in sandia_modules.columns:
    if 'Canadian_Solar' in mod:
        print(mod)
