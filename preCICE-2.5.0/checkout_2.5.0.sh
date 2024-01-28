#!/bin/bash
#
module load precice/2.5.0
cd ~/Workspace/PreCICE.jl
git checkout PreCICE.jl-v2.5.0
cd ~/Workspace/WaterLily/preCICE-2.5.0
preCICEJulia --project=~/Workscape/WaterLily/ -e "using Pkg; Pkg.build("PreCICE");"