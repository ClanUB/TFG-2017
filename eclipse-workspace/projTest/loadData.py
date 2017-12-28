import os
import numpy as np
import pandas as pd
from utilFunc import getMarks

def loadData(ensenyament):
    os.chdir("C:/Users/Orquidea/OneDrive/TFG-2017/pid-UB-master")
    data = {}
    id_enseny = "G1042"
    qual = pd.read_csv("qualifications_mates_info.csv", na_values = np.nan)
    qual = qual.drop(qual.columns[0], axis=1)
    
    data[id_enseny] = [qual]
    id_enseny = "G1077"
    data[id_enseny] = qual
    
    s1_info = [364288, 364289, 364290, 364291, 364292, 364293, 364294, 364298, 364299, 364301]
    s1_info_lbl = ['P1', 'DDB', 'IO', 'ALGE', 'CAL', 'MD', 'FIS', 'ALGO', 'P2', 'ED']
    
    s2_info = [364297, 364300, 364303, 364305, 364302, 364296, 364295, 364306, 364304, 364307]
    s2_info_lbl = ['ELEC', 'AA', 'DS', 'EC', 'ICC', 'EMP', 'PIE', 'PAE', 'PIS', 'SO1']
    
    s3_info = [364314, 364308, 364322, 364315, 364309, 364311, 364323, 364328, 364310, 364312]
    s3_info_lbl = ['IA', 'SO2', 'TNUI', 'VA', 'XAR', 'BD', 'FHIC', 'GiVD', 'LIL', 'SWD']
    
    s1_mates = [360142, 360140, 360136, 360138, 360134, 360135, 360139, 360143, 360137, 360141]
    s1_mates_lbl = ['ADIP', 'ELPR', 'IACD', 'LIRM', 'MAVE', 'ALLI', 'ARIT', 'FISI', 'IACI', 'PRCI']
    
    s2_mates = [360144, 360148, 360151, 360150, 360146, 360145, 360152, 360161, 360153, 360155]
    s2_mates_lbl = ['CDDV', 'ESAL', 'GELI', 'GRAF', 'MNU1', 'CIDV', 'GEPR', 'HIMA', 'MMSD', 'TOPO']
    
    s3_mates = [360158, 360149, 360156, 360147, 360162, 360159, 360154, 360163, 360160, 360157]
    s3_mates_lbl = ['ANMA', 'EQAL', 'GDCS', 'MNU2', 'PROB', 'ANCO', 'EQDI', 'ESTA', 'MODE', 'TGGS']
    
    s1_dret = [362441, 362442, 362444, 362451, 362446, 362443, 362452, 362449, 362450, 362447] 
    s1_dret_lbl = ['TTC', 'CP', 'FDD', 'DRO', 'PIC', 'EC', 'SDL', 'FDPTD', 'HD', 'DCP']
    
    s2_dret = [362448, 362453, 362454, 362456, 362459, 362461, 362469, 362458]
    s2_dret_lbl = ['OTE', 'PD', 'DOC', 'DIC', 'DFT', 'FDA', 'DPC', 'IDCE']
    
    s3_dret = [362507, 362460, 362462, 362466, 362465, 362470, 362467, 362463]
    s3_dret_lbl = ['DR', 'PST', 'CAA', 'DEM', 'DTS', 'DPP', 'DS', 'BPU']
    
    idAssig = np.concatenate((s1_info,s2_info,s3_info,s1_mates,s2_mates,s3_mates,s1_dret,s2_dret,s3_dret), axis=0)
    descAssig = np.concatenate((s1_info_lbl,s2_info_lbl,s3_info_lbl,s1_mates_lbl,s2_mates_lbl,s3_mates_lbl,s1_dret_lbl,s2_dret_lbl,s3_dret_lbl), axis=0)
    
    columns = dict(zip(idAssig,descAssig))
    qual = data[ensenyament][0]
    
    qual['id_assig'].replace(columns,inplace=True)
    qual['id_assig'] = qual['id_assig'][qual['id_assig'].isin(columns.values())]
    qual.dropna(axis = 0, how='any',subset=['id_assig'], inplace=True)
    
    if ensenyament == 'G1055':
        qual['nota'] = getMarks(qual)
        del qual['nota_primera_conv']
        del qual['nota_segona_conv']
    df = qual[qual['id_enseny'] == ensenyament]
    
    
    if ensenyament == "G1077":
        
        lbl1 = s1_info_lbl
        lbl2 = s2_info_lbl
        
    if ensenyament == "G1042":
        
        lbl1 = s1_mates_lbl
        lbl2 = s2_mates_lbl
    
    
    primer = pd.pivot_table(df,values = 'nota',index = 'id_alumne',columns='id_assig')[lbl1]
    segon = pd.pivot_table(df,values = 'nota',index = 'id_alumne',columns='id_assig')[lbl2]
    
    primer_segon = pd.concat([primer,segon],axis = 1)
    primer_segon = primer_segon.dropna(thresh = 11)
    primer_segon.drop(primer_segon[primer_segon.iloc[:,10:].sum(axis = 1) == 0].index,inplace=True)
    
    primer, segon = primer_segon.iloc[:,:10], primer_segon.iloc[:,10:20]
    return [primer, segon, lbl2]
    
