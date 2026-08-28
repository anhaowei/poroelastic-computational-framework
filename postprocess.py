#!/usr/bin/env python3
"""Generate cycle-10 paper figures and datasets from completed LV cases."""
import argparse, csv, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

MMHG=0.0075
COLORS=("black","#0072B2","#009E73","#CC79A7")

def need(p):
    p=Path(p).resolve()
    if not p.exists(): raise FileNotFoundError(p)
    return p

def numeric(p):
    rows=[]
    for line in need(p).open(errors="replace"):
        try: row=[float(x) for x in line.replace("\t",",").split(",")]
        except ValueError: continue
        if row and np.all(np.isfinite(row)): rows.append(row)
    if not rows: raise ValueError("No numeric data in {}".format(p))
    n=min(map(len,rows)); return np.asarray([x[:n] for x in rows])

def cycle_data(p,cycle,cols,scales,bcl):
    a=numeric(p); label=np.zeros(len(a),int)
    for i in range(1,len(a)): label[i]=label[i-1]+int(a[i,0]<a[i-1,0])
    pick=label==cycle-1
    if not np.any(pick) or a[pick,0].min()>2 or a[pick,0].max()<.99*bcl:
        raise ValueError("Complete cycle {} is absent in {}".format(cycle,p))
    return (a[pick,0],)+tuple(a[pick,c]*s for c,s in zip(cols,scales))

def pyplot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":18,"axes.labelsize":18,"xtick.labelsize":15,
      "ytick.labelsize":15,"legend.fontsize":12,"pdf.fonttype":42,"svg.fonttype":"none"})
    return plt

def save(fig,stem):
    for ext,kw in (("png",{"dpi":300}),("pdf",{}),("svg",{})): fig.savefig("{}.{}".format(stem,ext),**kw)

def table(p,rows):
    if rows:
        with Path(p).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def axis(ax,ylabel,bcl): ax.set(xlim=(0,bcl),xlabel="Time (ms)",ylabel=ylabel)

SERIES=(("a","output_Qcorin.txt",1,60000.,"Coronary inflow",r"$Q_{cor,in}$ (mL/min)"),
 ("b","output_PV.txt",2,MMHG,"LV cavity pressure","LV pressure (mmHg)"),
 ("c","output_porepres.txt",1,MMHG,"Average pore pressure","Pore pressure (mmHg)"),
 ("d","output_wallvolume.txt",1,1.,"LV wall volume",r"Wall volume (cm$^3$)"))

def figure3(case,cycle,bcl,out):
    plt=pyplot(); data=[]; rows=[]; summary=[]
    for panel,file,col,scale,title,ylabel in SERIES:
        t,y=cycle_data(case/file,cycle,(col,),(scale,),bcl); data.append((t,y,title,ylabel))
        rows += [{"panel":panel,"variable":title,"time_ms":x,"value":v} for x,v in zip(t,y)]
        summary.append({"panel":panel,"variable":title,"samples":len(t),"minimum":y.min(),"maximum":y.max(),"cycle_mean":np.trapz(y,t)/bcl})
        fig,ax=plt.subplots(figsize=(8,6)); ax.plot(t,y,"k",lw=1.8); axis(ax,ylabel,bcl); fig.tight_layout(); save(fig,out/("Figure3_cycle{}_baseline_{}".format(cycle,panel))); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for ax,(t,y,title,ylabel),panel in zip(axs.flat,data,"abcd"): ax.plot(t,y,"k"); axis(ax,ylabel,bcl); ax.set_title("({}) {}".format(panel,title),fontsize=13)
    fig.tight_layout(); save(fig,out/("Figure3_cycle{}_baseline_combined".format(cycle))); plt.close(fig)
    table(out/("Figure3_cycle{}_baseline_plot_data.csv".format(cycle)),rows); table(out/("Figure3_cycle{}_baseline_summary.csv".format(cycle)),summary)

def figure5(cases,cycle,bcl,out):
    plt=pyplot(); specs=(("a","output_Psin.txt",1,MMHG,"Coronary sinus pressure",r"$P_{sin}$ (mmHg)"),)+SERIES[0:1]+SERIES[2:]
    allcurves=[]; rows=[]; summary=[]
    for panel,file,col,scale,title,ylabel in specs:
        curves=[]
        for (r,case),color in zip(cases,COLORS):
            t,y=cycle_data(case/file,cycle,(col,),(scale,),bcl); curves.append((r,color,t,y))
            rows += [{"R_sin":r,"panel":panel,"variable":title,"time_ms":x,"value":v} for x,v in zip(t,y)]
            summary.append({"R_sin":r,"panel":panel,"variable":title,"minimum":y.min(),"maximum":y.max(),"cycle_mean":np.trapz(y,t)/bcl})
        allcurves.append(curves); fig,ax=plt.subplots(figsize=(8,6))
        for r,c,t,y in curves: ax.plot(t,y,color=c,label=r"$R_{sin}="+r+"$")
        axis(ax,ylabel,bcl); ax.legend(frameon=False); fig.tight_layout(); save(fig,out/("Figure5_cycle{}_CSO_{}".format(cycle,panel))); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for ax,spec,curves in zip(axs.flat,specs,allcurves):
        for r,c,t,y in curves: ax.plot(t,y,color=c,label=r)
        axis(ax,spec[5],bcl); ax.set_title("({}) {}".format(spec[0],spec[4]),fontsize=13)
    axs.flat[-1].legend(frameon=False); fig.tight_layout(); save(fig,out/("Figure5_cycle{}_CSO_combined".format(cycle))); plt.close(fig)
    table(out/("Figure5_cycle{}_CSO_plot_data.csv".format(cycle)),rows); table(out/("Figure5_cycle{}_CSO_summary.csv".format(cycle)),summary)

def pv(cases,cycle,bcl,out):
    plt=pyplot(); fig,ax=plt.subplots(figsize=(7.2,5.8)); rows=[]; stats=[]
    for (r,case),color in zip(cases,COLORS):
        t,v,p=cycle_data(case/"output_PV.txt",cycle,(1,2),(1.,MMHG),bcl); ax.plot(v,p,color=color,label=r"$R_{sin}="+r+"$")
        rows += [{"R_sin":r,"time_ms":x,"LV_volume_mL":a,"LV_pressure_mmHg":b} for x,a,b in zip(t,v,p)]
        edv,esv=v.max(),v.min(); stats.append({"R_sin":r,"EDV_mL":edv,"ESV_mL":esv,"SV_mL":edv-esv,"EF":(edv-esv)/edv,"peak_pressure_mmHg":p.max(),"stroke_work_mmHg_mL":abs(np.trapz(p,v))})
    ax.set(xlabel="LV cavity volume (mL)",ylabel="LV pressure (mmHg)"); ax.legend(frameon=False); fig.tight_layout(); save(fig,out/("lv_pv_loop_cycle{}_overlay".format(cycle))); plt.close(fig)
    table(out/("lv_pv_loop_cycle{}_data.csv".format(cycle)),rows); table(out/("lv_pv_loop_cycle{}_summary.csv".format(cycle)),stats)

def figure2(case,experiment,cycle,bcl,out):
    if not experiment:return
    plt=pyplot(); e=numeric(experiment); t,q=cycle_data(case/"output_Qcorin.txt",cycle,(1,),(60000.,),bcl)
    norm=lambda x:(x-x.min())/(x.max()-x.min()); en,sn=norm(e[:,1]),norm(q)
    fig,ax=plt.subplots(figsize=(8,6)); ax.plot(e[:,0],en,"r",label="Experiment"); ax.plot(t,sn,"k",label="Baseline simulation"); axis(ax,"Normalized coronary inflow",bcl); ax.legend(frameon=False); fig.tight_layout()
    stem=out/("Figure2_normalized_inflow_cycle{}".format(cycle)); save(fig,stem); plt.close(fig)
    table(str(stem)+"_plot_data.csv",([{"source":"experiment","time_ms":x,"raw_flow":y,"normalized_flow":z} for x,y,z in zip(e[:,0],e[:,1],en)]+[{"source":"simulation","time_ms":x,"raw_flow":y,"normalized_flow":z} for x,y,z in zip(t,q,sn)]))

def vtkmod():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
        return vtk,vtk_to_numpy,numpy_to_vtk
    except ImportError as e: raise RuntimeError("Run field processing with pvpython (VTK is required)") from e

def pvd(p):
    p=need(p); return sorted((float(n.attrib.get("timestep",0)),(p.parent/n.attrib["file"]).resolve()) for n in ET.parse(str(p)).getroot().findall(".//DataSet"))

def grid(p):
    vtk,_,_=vtkmod(); reader=vtk.vtkXMLPUnstructuredGridReader() if Path(p).suffix==".pvtu" else vtk.vtkXMLUnstructuredGridReader(); reader.SetFileName(str(p)); reader.Update(); g=vtk.vtkUnstructuredGrid(); g.DeepCopy(reader.GetOutput()); return g

def pressures(case,cycle,bcl,out):
    vtk,to_np,to_vtk=vtkmod(); start=(cycle-1)*bcl; entries=pvd(case/"pore_pressure_DG0.pvd"); targets=[start+x for x in (50,200,390,620)]
    raw=out/("cycle{}_pressure_four_times_raw".format(cycle)); normal=out/("cycle{}_pressure_four_times_normalized".format(cycle)); raw.mkdir(exist_ok=True); normal.mkdir(exist_ok=True); summary=[]
    collections=[]
    for directory in (raw,normal): collections.append([])
    for i,target in enumerate(targets):
        time,path=min(entries,key=lambda x:abs(x[0]-target)); g=grid(path); data=g.GetCellData(); a=data.GetArray("pore_pressure_DG0")
        if a is None:
            arrays=[data.GetArray(j) for j in range(data.GetNumberOfArrays()) if data.GetArray(j).GetNumberOfComponents()==1]
            if len(arrays)!=1: raise ValueError("Cannot identify pore-pressure cell array")
            a=arrays[0]
        values=to_np(a).astype(float); lo,hi=values.min(),values.max()
        for directory,normalized,k in ((raw,False,0),(normal,True,1)):
            gg=vtk.vtkUnstructuredGrid(); gg.DeepCopy(g)
            if normalized:
                x=to_vtk((values-lo)/(hi-lo) if hi>lo else np.zeros_like(values),deep=True); x.SetName("normalized_pore_pressure"); gg.GetCellData().AddArray(x); gg.GetCellData().SetActiveScalars(x.GetName())
            name="pressure_{:02d}.vtu".format(i); w=vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(str(directory/name)); w.SetInputData(gg); w.Write(); collections[k].append((time,name))
        summary.append({"requested_time_ms":target,"selected_time_ms":time,"minimum_model_units":lo,"maximum_model_units":hi,"minimum_mmHg":lo*MMHG,"maximum_mmHg":hi*MMHG})
    for directory,entries,name in ((raw,collections[0],"pressure_four_times.pvd"),(normal,collections[1],"pressure_four_times_normalized.pvd")):
        root=ET.Element("VTKFile",type="Collection",version="0.1",byte_order="LittleEndian"); c=ET.SubElement(root,"Collection")
        for t,f in entries: ET.SubElement(c,"DataSet",timestep=str(t),group="",part="0",file=f)
        ET.ElementTree(root).write(str(directory/name),encoding="utf-8",xml_declaration=True)
    table(normal/"normalization_summary.csv",summary); shutil.make_archive(str(raw),"zip",raw); shutil.make_archive(str(normal),"zip",normal)

def fields(case,cycle,bcl,out):
    """Create four-time pressure products and preserve solver regional diagnostics."""
    pressures(case,cycle,bcl,out)
    source=need(case/"regional_transmural_balance.csv"); rows=[]
    with source.open(newline="") as f:
        for r in csv.DictReader(f):
            if int(float(r["cycle"]))==cycle-1: rows.append(r)
    table(out/("Figure4b_cycle{}_regional_transmural_balance.csv".format(cycle)),rows)

def case_arg(s):
    if "=" not in s: raise argparse.ArgumentTypeError("use R_sin=directory")
    r,p=s.split("=",1); return r,need(p)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--baseline-case",type=Path,required=True); p.add_argument("--cso-case",action="append",type=case_arg,default=[],metavar="R_SIN=DIR"); p.add_argument("--experiment-csv",type=Path); p.add_argument("--cycle",type=int,default=10); p.add_argument("--bcl",type=float,default=660.); p.add_argument("--result-root",type=Path,default=Path(__file__).resolve().parent/"outputs"); p.add_argument("--skip-fields",action="store_true"); a=p.parse_args()
    if a.cycle<1:p.error("--cycle must be positive")
    baseline=need(a.baseline_case); cases=[("100",baseline)]+a.cso_case
    if len(cases)!=4:p.error("provide three --cso-case values for 1e4, 1e5 and 1e6")
    plots=a.result_root.resolve()/("cycle{}_baseline_figure3_style_v1".format(a.cycle)); field=a.result_root.resolve()/("cycle{}_P50_35_midwall_paper_postprocessing_v1".format(a.cycle)); plots.mkdir(parents=True,exist_ok=True); field.mkdir(parents=True,exist_ok=True)
    figure2(baseline,a.experiment_csv,a.cycle,a.bcl,plots); figure3(baseline,a.cycle,a.bcl,plots); figure5(cases,a.cycle,a.bcl,plots); pv(cases,a.cycle,a.bcl,plots)
    if not a.skip_fields: fields(baseline,a.cycle,a.bcl,field)
    print(plots); print(field)
if __name__=="__main__":main()
