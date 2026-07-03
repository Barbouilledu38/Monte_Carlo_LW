#!/bin/sh -e

# Copyright (C) 2018, 2020, 2021, 2023 |Méso|Star> (contact@meso-star.com)
# Copyright (C) 2018 Centre National de la Recherche Scientifique
# Copyright (C) 2018 Université Paul Sabatier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

path_Simu=/home/barroisl/edstar/Simus
path_transect=/home/barroisl/Transect_MC_auto

mie_lut="${path_Simu}/Mie_LUT_Cloud-2-10-0.010.nc"
clouds="${path_Simu}/clouds/L12km.1.BOMEX.005.htcp"
#materials="${path_Simu}/materials/chartreuse_thomas.mtls"

image_def="15x15"
image_spp="50"
image="def=${image_def}:spp=${image_spp}"

opthick="1"
cache="L12km${opthick}.cache"

sun_dir="10,10"

output="./chartreuse_${image_def}x${image_spp}"

export HTRDR_ATMOSPHERE_SPK="${path_Simu}"

for atm in "${path_Simu}/atms"/*; do

    atmosphere="${path_Simu}/atms/ecrad_opt_prop_${atm: -11:7}.txt"

    for topo  in "${path_Simu}/models/fake_2/canyon_0.5_"*; do

        dist="${topo: -8:4}"
        directory="/home/barroisl/Transect_MC_auto/Data/fake_2/${atm: -11:7}/${dist}"
        
        echo "$dist"

        if [ ! -d "$directory" ]; then
            mkdir $directory
        else 
            rm -f $directory/*
        fi

        ground="${path_Simu}/models/fake_2/canyon_0.5_${dist}.obj"
        materials="${path_Simu}/materials/fake_2/canyon_0.5_${dist}.mtls"

        position="pos=50000,50000,0"
        tgt="tgt=50000,50000,100"
        sz="sz=0.1,0.1"
        up="up=0,1,0"
        cam=${position}:${tgt}:${up}:${sz}

        set -x
        htrdr-atmosphere -v\
            -a "${atmosphere}"\
            -g "${ground}" -R\
            -M "${materials}"\
            -D "${sun_dir}" \
            -i "${image}"\
            -l \
            -s lw=4000,40000:Tref=275 \
            -p "${cam}" \
            -T "${opthick}"\
            -O "${cache}"\
            -fo "${output}.txt"

        mv ${path_Simu}/output_path_*.txt $directory/

        for f in $directory/output_path_*.txt; do

            tail -n +2 "$f"

        done > $directory/${line_number}_50_15_15.txt

        #rm -f $directory/output_*.txt
    done
done
