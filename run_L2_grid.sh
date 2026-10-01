#!/bin/bash
# Seviye 2 izgara: 4 kil yuku sistemi (dizustu, seri)
# her sistem: delete_atoms ile uret -> dengele -> adhezyon olc

declare -A SIL=( ["50"]="none" ["37"]="122:134" ["25"]="110:134" ["13"]="98:134" )
echo "# zincir  adhezyon_kcalmol  atom" > L2_grid_results.txt

for N in 50 37 25 13; do
  echo "=== $N zincir sistemi ==="
  # 1) sistem uret (50 = referans, silme yok)
  if [ "${SIL[$N]}" == "none" ]; then
    cp interface_equil.data interface_L2_${N}.data
  else
    cat > _make_${N}.inp <<ZZZ
units real
atom_style full
boundary p p f
pair_style lj/charmmfsw/coul/long 10 12
pair_modify mix arithmetic
bond_style harmonic
angle_style charmm
dihedral_style charmmfsw
improper_style harmonic
special_bonds charmm
kspace_style pppm 1e-6
kspace_modify slab 3.0
read_data interface_equil.data
group kaldir molecule ${SIL[$N]}
delete_atoms group kaldir bond yes mol yes
write_data interface_L2_${N}.data
ZZZ
    conda run -n md lmp -in _make_${N}.inp > _make_${N}.log 2>&1
  fi

  # 2) dengele + adhezyon olc
  cat > _meas_${N}.inp <<ZZZ
units real
atom_style full
boundary p p f
pair_style lj/charmmfsw/coul/long 10 12
pair_modify mix arithmetic
bond_style harmonic
angle_style charmm
dihedral_style charmmfsw
improper_style harmonic
special_bonds charmm
kspace_style pppm 1e-6
kspace_modify slab 3.0
processors * * 1
read_data interface_L2_${N}.data
group clay type 1:16
group pva type 17:24
neighbor 2.0 bin
neigh_modify delay 5 every 1 check yes
velocity all create 300.0 4321 dist gaussian loop geom
fix 1 all nvt temp 300.0 300.0 100.0
run 10000
reset_timestep 0
compute adh clay group/group pva pair yes kspace yes
variable adhv equal c_adh
fix avea all ave/time 100 100 10000 v_adhv
run 10000
variable Aout equal f_avea
print "GRID ${N} \${Aout}"
ZZZ
  conda run -n md mpirun -np 10 lmp -in _meas_${N}.inp > _meas_${N}.log 2>&1
  A=$(grep "GRID ${N}" _meas_${N}.log | awk '{print $3}')
  NAT=$(grep " atoms" interface_L2_${N}.data | head -1 | awk '{print $1}')
  echo "$N  $A  $NAT" | tee -a L2_grid_results.txt
done

echo ""
echo "=== IZGARA TAMAMLANDI ==="
cat L2_grid_results.txt
