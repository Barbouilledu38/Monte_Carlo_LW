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

atmosphere="${path_Simu}/atms/ecrad_opt_prop_TUXUATM.txt"
mie_lut="${path_Simu}/Mie_LUT_Cloud-2-10-0.010.nc"
clouds="${path_Simu}/clouds/L12km.1.BOMEX.005.htcp"
materials="${path_Simu}/materials/chartreuse_thomas.mtls"

image_def="15x15"
image_spp="50"
image="def=${image_def}:spp=${image_spp}"

opthick="1"
cache="L12km${opthick}.cache"

sun_dir="10,10"

output="./chartreuse_${image_def}x${image_spp}"

export HTRDR_ATMOSPHERE_SPK="${path_Simu}"

altitude_snow=800
materials="${path_Simu}/materials/chartreuse_thomas.mtls"
ground="${path_Simu}/models/chartreuse_thomas.obj"

line_number=0
while IFS= read -r line; do
    ((line_number++))
    directory=${path_transect}/Data/guiers_250_144_atms/TUXUATM/$line_number
    if [ ! -d "$directory" ]; then
        mkdir "$directory"
    else
        rm -rf $directory/*
    fi

    set -- $line 

    echo "line" $line
    x_pos=$1
    y_pos=$2
    z_pos=$3

    x_tgt=$4
    y_tgt=$5
    z_tgt=$6

    set -- $line
    echo $0

    position="pos=$1,$2,$3"
    tgt="tgt=$4,$5,$6"
    sz="sz=2,2"
    up="up=0,1,0"
    cam=${position}:${tgt}:${up}:${sz}

    echo -e "$x_pos\n$y_pos\n$z_pos\n$x_tgt\n$y_tgt\n$z_tgt\n" >> ${directory}/${line_number}_camera_tgt.txt
    
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

done < ${path_transect}/camera_tgt/polygon_guiers.txt


