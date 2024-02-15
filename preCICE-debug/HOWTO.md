## Install the preCICE library

set the correct version of the preCICE library in the precice repo
```bash
cd /apps/precice/precice
git checkout v3.0.0
```
go in the `build` directory and run the following commands 

```bash
cd /apps/precice/build
make clean
```

Now we are ready to build the version we want

```bash
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/apps/precice/3.0.0 ../precice/
sudo make -j $nproc 
sudo make install
```

## Run the simulations

We need to load the correct preCICE version
    
```bash
module load preCICE/3.0.0
```

make suer the version of the `PreCICE.jl` is the same as the binary

```bash
cd /apps/precice/PreCICE.jl
git checkout v3.0.0
```

see the function that are in the `preCICE` library

```bash
nm -D /apps/precice/3.0.0/lib/libprecice.so | grep precicec
```

## G+Smo

```bash
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/apps/gismo/stable ../gismo/
sudo make -j 4
sudo make install
```

```bash
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/apps/gismo/stable -DCMAKE_PREFIX_PATH=/apps/precice/3.0.0/lib/cmake/precice ../gismo/
```