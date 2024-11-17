import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CathDomain:
    domain_name: str          # CATH domain name
    class_number: int         # Class number
    architecture: int         # Architecture number
    topology: int             # Topology number
    homologous_superfamily: int # Homologous superfamily number
    s35: int                  # S35 sequence cluster number
    s60: int                  # S60 sequence cluster number
    s95: int                  # S95 sequence cluster number
    s100: int                 # S100 sequence cluster number
    s100_count: int           # S100 sequence count number
    domain_length: int        # Domain length
    resolution: float         # Structure resolution in Angstroms

def parse_cath_file(filepath: str) -> List[CathDomain]:
    domains = []
    
    with open(filepath, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue

            # Parse the line into the respective fields
            fields = line.split()
            if len(fields) < 12:
                print(f"Skipping malformed line: {line.strip()}")
                continue
            
            # Convert and assign fields according to column definitions
            domain = CathDomain(
                domain_name=fields[0],
                class_number=int(fields[1]),
                architecture=int(fields[2]),
                topology=int(fields[3]),
                homologous_superfamily=int(fields[4]),
                s35=int(fields[5]),
                s60=int(fields[6]),
                s95=int(fields[7]),
                s100=int(fields[8]),
                s100_count=int(fields[9]),
                domain_length=int(fields[10]),
                resolution=float(fields[11])
            )
            domains.append(domain)
    
    return domains

