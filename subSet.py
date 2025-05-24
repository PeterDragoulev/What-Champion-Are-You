import os, shutil, pathlib, math


def __make_subset__():
    original_dir = pathlib.Path("temp")

    new_base_dir = pathlib.Path("league_champs_small")

    for category in ("Lux", "Akali", "MissFortune", "Ahri", "Caitlyn", "Ezreal", "Blitzcrank", "LeeSin", "TwistedFate",
                     "Draven"): 
        dir_store = new_base_dir / "validation" / category
        os.makedirs(dir_store)

        dir_store = new_base_dir / "train" / category
        os.makedirs(dir_store)

        dir_fetch = original_dir / category
        onlyFiles = [f for f in dir_fetch.iterdir() if f.is_file()]

        fnames = [f"{category}.{i}"
                  for i in onlyFiles]

        total = len(fnames)
        valid = math.ceil(total*0.2)
        train = total - valid

        c = 0
        for f in onlyFiles:
            new_name = f"{category}.{f.name}"
            if c >= train:
                dir_store = new_base_dir / "validation" / category
                shutil.copyfile(src=f, dst=dir_store / new_name)
            else:
                dir_store = new_base_dir / "train" / category
                shutil.copyfile(src=f, dst=dir_store / new_name)
            c += 1

def makeSubset():
    if pathlib.Path("league_champs_small").exists() and pathlib.Path("league_champs_small").is_dir():
        shutil.rmtree(pathlib.Path("league_champs_small"))

    __make_subset__()