"""
PVA(-OH)...kil(O) arayuz hidrojen baglarini sayar.
Kriter: O(PVA)...O(kil) < 3.5 A  VE  aci O-H...O(kil) > 120 derece.
Girdi:  interface_equil.data (topoloji) + interface.lammpstrj (trajektori)
Cikti:  konsola ortalama H-bagi sayisi + bagli PVA-OH orani
Calistir (numpy olan ortamda, ornegin mobo; ~/CLAY/interface klasorunde):
    python hbond.py
"""
import numpy as np

DATA = "interface_equil.data"
DUMP = "interface.lammpstrj"
R_CUT = 3.5     # O...O mesafe esigi (A)
ANG_CUT = 120.0  # O-H...O aci esigi (derece)
CLAY_TYPES = set(range(1, 17))    # kil atom tipleri
PVA_TYPES = set(range(17, 25))    # PVA atom tipleri

# ---------- 1) data dosyasindan tip/kutle/bag oku ----------
def parse_data(path):
    lines = open(path).read().split("\n")
    masses, types, bonds = {}, {}, []
    sec = None
    for ln in lines:
        s = ln.strip()
        if s in ("Masses", "Atoms", "Bonds") or s.startswith(("Atoms ", "Masses ", "Bonds ")):
            sec = s.split()[0]; continue
        if s == "" or s.startswith(("Pair", "Bond Coeffs", "Angle", "Dihedral",
                                    "Velocities", "Improper")):
            if s.startswith(("Pair", "Angle", "Dihedral", "Velocities", "Improper",
                             "Bond Coeffs")):
                sec = None
            continue
        p = s.split()
        if sec == "Masses" and len(p) >= 2 and p[0].isdigit():
            masses[int(p[0])] = float(p[1])
        elif sec == "Atoms" and len(p) >= 7 and p[0].isdigit():
            types[int(p[0])] = int(p[2])
        elif sec == "Bonds" and len(p) >= 4 and p[0].isdigit():
            bonds.append((int(p[2]), int(p[3])))
    return masses, types, bonds

masses, atype, bonds = parse_data(DATA)

def elem(t):
    m = masses[t]
    for e, mm in [("H",1), ("C",12), ("O",16), ("Na",23), ("Mg",24),
                  ("Al",27), ("Si",28)]:
        if abs(m - mm) < 1.5:
            return e
    return "?"

pva_O = {t for t in PVA_TYPES if elem(t) == "O"}   # PVA hidroksil O (donor)
pva_H = {t for t in PVA_TYPES if elem(t) == "H"}   # PVA H
clay_O = {t for t in CLAY_TYPES if elem(t) == "O"}  # kil O (acceptor)
print(f"PVA O tipleri (donor)  : {sorted(pva_O)}")
print(f"clay O tipleri (accept): {sorted(clay_O)}")

# hidroksil O-H ciftleri: PVA-O'ya bagli PVA-H
o_atoms = {i for i, t in atype.items() if t in pva_O}
h_by_o = {}
for a, b in bonds:
    if a in o_atoms and atype.get(b) in pva_H: h_by_o[a] = b
    elif b in o_atoms and atype.get(a) in pva_H: h_by_o[b] = a
donors = [(o, h) for o, h in h_by_o.items()]        # (O_donor, H)
donor_O = np.array([o for o, h in donors])
donor_H = np.array([h for o, h in donors])
accept = np.array([i for i, t in atype.items() if t in clay_O])
print(f"PVA hidroksil grubu    : {len(donors)}")
print(f"kil O (acceptor) atomu : {len(accept)}")

# ---------- 2) trajektori karelerini oku ----------
def frames(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line: break
            if "TIMESTEP" in line:
                ts = int(f.readline())
            elif "NUMBER OF ATOMS" in line:
                n = int(f.readline())
            elif "BOX BOUNDS" in line:
                xlo, xhi = map(float, f.readline().split())
                ylo, yhi = map(float, f.readline().split())
                zlo, zhi = map(float, f.readline().split())
                box = (xhi-xlo, yhi-ylo)
            elif "ITEM: ATOMS" in line:
                coord = {}
                for _ in range(n):
                    p = f.readline().split()
                    coord[int(p[0])] = (float(p[2]), float(p[3]), float(p[4]))
                yield ts, box, coord

def mic(d, L):   # minimum image (x,y periyodik)
    return d - L*np.round(d/L)

# ---------- 3) her karede H-bagi say ----------
counts = []
for ts, (Lx, Ly), coord in frames(DUMP):
    Od = np.array([coord[i] for i in donor_O])
    Hd = np.array([coord[i] for i in donor_H])
    Oa = np.array([coord[i] for i in accept])
    nb = 0
    for k in range(len(Od)):
        dx = mic(Oa[:,0]-Od[k,0], Lx)
        dy = mic(Oa[:,1]-Od[k,1], Ly)
        dz = Oa[:,2]-Od[k,2]
        r = np.sqrt(dx*dx+dy*dy+dz*dz)
        near = np.where(r < R_CUT)[0]
        for j in near:
            # aci O_d - H - O_a
            v1 = np.array([mic(Od[k,0]-Hd[k,0],Lx), mic(Od[k,1]-Hd[k,1],Ly), Od[k,2]-Hd[k,2]])
            v2 = np.array([mic(Oa[j,0]-Hd[k,0],Lx), mic(Oa[j,1]-Hd[k,1],Ly), Oa[j,2]-Hd[k,2]])
            cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
            ang = np.degrees(np.arccos(np.clip(cosang,-1,1)))
            if ang > ANG_CUT:
                nb += 1
    counts.append(nb)

counts = np.array(counts)
print("\n==== ARAYUZ HIDROJEN BAGI ====")
print(f"kare sayisi           : {len(counts)}")
print(f"H-bagi (ortalama)     : {counts.mean():.1f} +/- {counts.std():.1f}")
print(f"bagli PVA-OH orani    : %{100*counts.mean()/len(donors):.1f}")
print(f"H-bagi / nm^2 yuzey   : {counts.mean()/(Lx*Ly/100):.2f}")
