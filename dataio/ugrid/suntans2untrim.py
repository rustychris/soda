"""
Tool for converting from a suntans to untrim netcdf formats
"""
from ..suntans.sunpy import Spatial, Grid
from ...utils import othertime
from datetime import datetime
from netCDF4 import Dataset
import numpy as np
import glob
import os
import os.path
from .untrim_tools import untrim_ugrid as ugrid

import pdb

#Dictionary containing the suntans-untrim equivalent grid variables
untrim_gridvars = {\
        'xp':'Mesh2_node_x',\
        'yp':'Mesh2_node_y',\
        'xv':'Mesh2_face_x',\
        'yv':'Mesh2_face_y',\
        'xe':'Mesh2_edge_x',\
        'ye':'Mesh2_edge_y',\
        'mark':'Mesh2_edge_bc',\
        'edges':'Mesh2_edge_nodes',\
        'grad':'Mesh2_edge_faces',\
        'cells':'Mesh2_face_nodes',\
        'face':'Mesh2_face_edges',\
        'dv':'Mesh2_face_depth',\
        'de':'Mesh2_edge_depth',\
        'z_r':'Mesh2_layer_3d',\
        'time':'Mesh2_data_time',\
        'mark':'Mesh2_edge_bc',\
        'facemark':'Mesh2_face_bc'\
    }

# Dictionary containing the suntans-untrim  equivalent grid dimensions
untrim_griddims = {\
        'Np':'nMesh2_node',\
        'Ne':'nMesh2_edge',\
        'Nc':'nMesh2_face',\
        'Nkmax':'nMesh2_layer_3d',\
        'numsides':'nMaxMesh2_face_nodes',\
        'time':'nMesh2_data_time'\
    }

# Dimensions with hard-wired values
other_dims = {\
        'Three':3,\
        'Two':2,\
        'nsp':1,\
        'date_string_length':19,\
        'nMesh2_time':1\
        }

# physical variables that are directly transferable
untrim_vars = {\
        'salt':'Mesh2_salinity_3d',\
        'nu_v':'Mesh2_vertical_diffusivity',\
        'eta':'Mesh2_sea_surface_elevation'
        }

varnames = ['Mesh2_salinity_3d',\
            'Mesh2_vertical_diffusivity_3d',\
            'Mesh2_sea_surface_elevation',\
            'h_flow_avg',\
            'v_flow_avg',\
            'Mesh2_edge_wet_area',\
            'Mesh2_face_wet_area',\
            'Mesh2_edge_bottom_layer',\
            'Mesh2_edge_top_layer',\
            'Mesh2_face_bottom_layer',\
            'Mesh2_face_top_layer',\
            'Mesh2_face_water_volume',\
            ]

FILLVALUE=-9999

@profile
def suntans2untrim(ncfile,outfile,tstart,tend,grdfile=None,
                   dzmin=0.001):
    """
    Converts a suntans averages netcdf file into untrim format
    for use in particle tracking
    """
    ####
    # Step 1: Load the suntans data object
    ####
    sun = Spatial(ncfile,klayer=[-99])

    # Calculate some other variables
    sun.de = sun.get_edgevar(sun.dv,method='min')
    sun.mark[sun.mark==5]=0
    # This seems to lose flow vs. open information.
    # sun.mark[sun.mark==3]=2

    sun.facemark = sun.get_facemark()
    # This will come back with polygons adjacent to closed boundary
    # set to 1, but for PTM I think that should be 0.
    sun.facemark[ sun.facemark==1 ] = 0

    # Update the grad variable from the ascii grid file if supplied
    if grdfile is not None:
        print('Updating grid with ascii values...')
        grd = Grid(grdfile)
        sun.grad = grd.grad[:,::-1]

    ###
    # Step 2: Write the grid variables to a netcdf file
    ###
    if os.path.exists(outfile): os.unlink(outfile)
    
    nc = Dataset(outfile,'w',format='NETCDF4_CLASSIC')

    # Global variable
    nc.Description = 'UnTRIM history file converted from SUNTANS output'

    # Write the dimensions
    for dd in list(untrim_griddims.keys()):
        if dd == 'time':
            nc.createDimension(untrim_griddims[dd],0)
        elif dd =='numsides':
            nc.createDimension(untrim_griddims[dd],sun.maxfaces)
        else:
            nc.createDimension(untrim_griddims[dd],sun[dd])

    for dd in other_dims:
        nc.createDimension(dd,other_dims[dd])

    ###
    # Step 3: Initialize all of the grid variables
    ###
    def create_nc_var(name, dimensions, attdict,data=None, 
                      dtype='f8',zlib=False,complevel=0,fill_value=None):

        tmp=nc.createVariable(name, dtype, dimensions,\
            zlib=zlib,complevel=complevel,fill_value=fill_value)

        for aa in list(attdict.keys()):
            tmp.setncattr(aa,attdict[aa])

        if data is not None:
            nc.variables[name][:] = data

    # Make sure the masked cells have a value of -1
    mask = sun['cells'].mask.copy()
    sun['cells'][mask]=FILLVALUE
    sun['face'][mask]=FILLVALUE

    for vv in list(untrim_gridvars.keys()):
        vname = untrim_gridvars[vv]
        print('Writing grid variable %s (%s)...'%(vname,vv))

        if vv=='time':
            continue

        # add dz_min attribute to z_r variable
        if vv == 'z_r':
            # This used to default to 1e-5. Untrim uses 1e-3 (at least that's
            # what's in the output file).  This value is interpreted most like
            # dzmin_surface of the suntans code.
            ugrid[vname]['attributes'].update({'dz_min':dzmin})
            #sun[vv][:]=sun[vv][::-1]
            sun[vv][:]=sun['z_w'][0:-1][::-1]

        # Reverse the order of grad(???)
        if vv=='grad':
            # sun[vv][:]=sun[vv][:,::-1]
            # RH this appears to be causing problems in FISH_PTM.
            # trying without this...
            print("RH: leaving grad order alone")

        ## Fix one-based indexing
        #if vv in ['cells','edges','grad']:
        #    mask = sun[vv][:]==-1
        #    tmp = sun[vv][:]+1
        #    tmp[mask]=-1
        #    #sun[vv][:]=sun[vv][:]+1
        #    create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
        #        data=tmp,dtype=ugrid[vname]['dtype'])

        create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
            data=sun[vv],dtype=ugrid[vname]['dtype'])


    # Initialize the two time variables
    vname=untrim_gridvars['time']
    create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
            dtype=ugrid[vname]['dtype'])
    vname = 'Mesh2_data_time_string'
    create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
            dtype=ugrid[vname]['dtype'])

    ###
    # Step 4: Initialize all of the time-varying variables (but don't write)
    ###
    for vname  in varnames:
        print('Creating variable %s...'%(vname))

        # used to set fill value, but that complicates some debugging, and departs
        # from what untrim does.
        create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
            dtype=ugrid[vname]['dtype'],zlib=True,complevel=1)

    ###
    # Step 5: Loop through all of the time steps and write the variables
    ###
    if tstart is None:
        tstart=sun.time[0]
    if tend is None:
        tend=sun.time[-1]
    tsteps=sun.getTstep(tstart,tend)

    # precompute the neighbor array
    e2c_mirror=nc['Mesh2_edge_faces'][:,:].copy()
    bound= e2c_mirror.min(axis=1)<0 # order agnostic 
    e2c_mirror[bound,:] = e2c_mirror[bound,:].max(axis=1)[:,None]
    # not used, but reported to help with debugging -- this is the
    # running estimate of dzmin_surface
    dzmin_surface_estimate=0.0
    
    tdays = othertime.DaysSince(sun.time,basetime=datetime(1899,12,31))
    for ii, tt in enumerate(tsteps):
        # Convert the time to the untrim formats
        timestr = datetime.strftime(sun.time[tt],'%Y-%m-%d %H:%M:%S')

        print('Writing data at time %s (%d of %d)...'%(timestr,tt,tsteps[-1]))

        #Write the time variables
        nc.variables['Mesh2_data_time'][ii]=tdays[ii]
        nc.variables['Mesh2_data_time_string'][:,ii]=timestr

        # Load each variable or calculate it and convert it to the untrim format
        sun.tstep=[tt]

        ###
        # Compute a few terms first
        eta = sun.loadData(variable='eta' )

        if ii<2:
            # true average not defined for ii=0
            # not sure why still nan at ii=1
            eta_avg=eta
        else:
            eta_avg = sun.loadData(variable='eta_avg')
        U = sun.loadData(variable='U_F' )
        dzz = sun.getdzz(eta)
        assert np.all(dzz>=0)
        # Note that this etop and ctop do *not* reflect any dzmin
        dzf,etop = sun.getdzf(eta_avg,return_etop=True)
        ctop = sun.loadData(variable='ctop')
        assert np.all(dzf>=0)

        # Use nu_v as a marker for ctop according to suntans, which *does*
        # include dzmin_surface.  Doesn't work for ii=0, though
        etop_sun=etop.copy()
        ctop_sun=ctop.copy()
        if ii>0: # nu_v is all zero on first time step
            nu_v=sun.loadData(variable='nu_v')
            wet_cells=(~nu_v.mask) & (nu_v.data>0.0)
            C=np.arange(sun.Nc)
            # Lumped surface cells:
            # nu_v doesn't agree with ctop,
            # ctop shows multiple layers
            # and nu_v shows at least 1 wet layer.
            lumped=(~wet_cells[ctop_sun,C])&(ctop_sun<sun.Nk)&(wet_cells.sum(axis=0)>0)
            # specifically, cells that are not lumped but could be
            # just used to get some bounds on dzmin_surface
            not_lumped=(wet_cells[ctop_sun,C])&(ctop_sun<sun.Nk)&(wet_cells.sum(axis=0)>0)

            if np.any(lumped):
                dzz_lumped_max=dzz[ctop[lumped],C[lumped]].max()
                dzmin_surface_estimate=max(dzmin_surface_estimate,dzz_lumped_max)
                print("dzz_lumped_max: %.4f  dzmin_surface ~ %.4f"%(dzz_lumped_max,dzmin_surface_estimate))

                for i in np.nonzero(lumped)[0]:
                    k_wet=np.nonzero(wet_cells[:,i])[0]
                    ctop_sun[i]=k_wet[0]
            # And adjust etop to show the same number of layers as the upwinded
            # ctop
            eta_nbr=eta[e2c_mirror]
            ctop_up=np.where( eta_nbr[:,0]>=eta_nbr[:,1],
                              ctop_sun[e2c_mirror[:,0]],
                              ctop_sun[e2c_mirror[:,1]] )
            etop_sun=np.minimum(ctop_up,sun.Nke-1)

            n_etop_changed=(etop_sun!=etop).sum()
            print("Surface lumping changed %d ctops"%( (ctop!=ctop_sun).sum() ))
            print("Surface lumping changed %d etops"%n_etop_changed)
            
        vname='Mesh2_sea_surface_elevation'
        #print '\tVariable: %s...'%vname
        nc.variables[vname][:,ii]=eta

        # UnTRIM references from bottom to top i.e.
        # k = 0 @ bed ; k = Nkmax-1 @ top
        # but the indices are expected to be 1-based, and top is inclusive
        # [per ESG, 2020-04-14]

        vname = 'Mesh2_edge_bottom_layer'
        #print '\tVariable: %s...'%vname
        kbj=sun.Nkmax-sun.Nke+1 # one based
        nc.variables[vname][:,ii]=kbj

        vname = 'Mesh2_edge_top_layer'
        #print '\tVariable: %s...'%vname
        ktj = sun.Nkmax-etop # one based, but inclusive, but >=kbj.
        dry=ktj<kbj
        # This is okay even though *most* edges that are dry in untrim get
        # ktj=0.  PTM just props the value up to kbj anyway.
        ktj[dry]=kbj[dry]
        nc.variables[vname][:,ii]=ktj
        
        vname = 'Mesh2_edge_wet_area'
        #print '\tVariable: %s...'%vname
        #dzf = sun.loadData(variable='dzf')
        tmp3d = dzf*sun.df
        assert np.all(tmp3d>=0.0)
        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]
        
        vname = 'Mesh2_salinity_3d'
        #print '\tVariable: %s...'%vname
        tmp3d = sun.loadData(variable='salt' )
        if tmp3d is None:
            val=0.0
        else:
            if sun.Nkmax==1:
                tmp3d=tmp3d[None,:]
            val=tmp3d.swapaxes(0,1)[:,::-1]
        nc.variables[vname][:,:,ii]=val

        vname = 'Mesh2_vertical_diffusivity_3d'
        #print '\tVariable: %s...'%vname
        tmp3d = sun.loadData(variable='nu_v' )
        if sun.Nkmax==1:
            tmp3d=tmp3d[None,:]
        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]

        vname = 'h_flow_avg'
        #print '\tVariable: %s...'%vname
        # RH: in the past this code flipped grad, and assigned U as-is.
        #  better to flip the sign here, but leave grad alone so that boundary
        #  edges do not get a -1 in the first index.
        if sun.Nkmax==1:
            U=U[None,:]
        U=np.ma.filled(U,0.0) # more consistent with untrim output

        for j in range(sun.Ne):
            #,ktop in enumerate(np.minimum(etop,sun.Nke-1)):
            ktop=etop[j]
            ktop_sun=etop_sun[j]
            assert ktop <= sun.Nke[j]-1
            assert ktop_sun <= sun.Nke[j]-1

            # Because U reflects integrated flow, but etop is from eta_avg
            # there can be some U above etop, which should be collapsed down
            # to etop
            U[ktop,j]+=U[:ktop,j].sum()
            U[:ktop,j]=0.0

            # Due to surface layer lumping with dzmin_surface, we can also
            # have the opposite problem, where U is zero in the top layer
            if ktop_sun>ktop:
                # use dzf to redistribute the flow
                weights=dzf[ktop:ktop_sun+1,j]
                # This shouldn't happen since etop and dzf are computed at the
                # same time above.
                assert np.all(weights>0.0),"Confusing etop with no dzf"
                weights /= weights.sum()
                U[ktop:ktop_sun+1,j] = weights*U[ktop_sun,j]
            
        # HERE - ideally we'd have suntans data on the average flux area, so 
        # we'd have direct evidence of which layers were active and which were
        # not.  then if etop was above the active layers, we know that dzmin had
        # caused some lumping, and we can unlump here.
        # unfortunately that data is not in the suntans output.
        # can possibly use nu_v to determine which cells are active.
        # this appears to be valid
        # Maybe try out using this to determine dzmin??
        #   more to make sure I'm getting this right?
        # Maybe not strictly necessary. but note this doesn't directly get
        # us the correction for edges -- still have to do something like upwind
        # eta, or use next velocity down to upwind eta, then grab the ctop of that
        # upwinded cell.
        # I think that's the way to go. Later, optionally, can re-apply a new dzmin.
        # may not be strictly necessary, just a minor loss of volume.(!)
        nc.variables[vname][:,:,ii]=-U.swapaxes(0,1)[:,::-1]

        vname = 'v_flow_avg'
        #print '\tVariable: %s...'%vname
        tmp3d = sun.loadData(variable='w' ) * sun.Ac # m^3/s
        if sun.Nkmax==1:
            tmp3d=tmp3d[None,:]
        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]

        # Need to calculate a few terms for the other variables

        vname = 'Mesh2_face_water_volume'
        #print '\tVariable: %s...'%vname
        #dzz = sun.loadData(variable='dzz')
        tmp3d = dzz*sun.Ac
        assert np.all(dzz>=0)
        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]

        vname = 'Mesh2_face_wet_area'
        #print '\tVariable: %s...'%vname
        tmp3d = np.repeat(sun.Ac[np.newaxis,...],sun.Nkmax,axis=0)
        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]

        if 0:# paranoid testing:
            # array() to drop the mask
            kbj0=np.array(kbj)-1 # 0-based index of deepest FV above the bed.
            ktj0=np.array(ktj)-1 # 0-based index of upper-most wet FV.
            n_j=nc.dimensions['nMesh2_edge'].size
            Q_j=np.ma.filled(nc.variables['h_flow_avg'][:,:,ii], np.nan)
            for j in range(n_j):
                # NOTE! Mesh2_edge_wet_area can have an extra layer on top.
                Q=Q_j[j,:]
                assert (kbj0[j]==0) or np.all( Q[:kbj0[j]]==0 )
                # assert np.all( ~Q[kbj0[j]:ktj0[j]+1].mask )
                # assert np.all( ~Q[kbj0[j]:].mask )
                above_surface=Q[ktj0[j]+1:]
                assert np.all( np.isnan(above_surface)|(above_surface==0.0) )
                
        vname = 'Mesh2_face_bottom_layer'
        # This is confusing because Nk has a different meaning.  It has a max
        # of 49, one less than Nkmax or the max of Nke.
        # It's an oddity in sunpy -- it is explicity decremented to reflect the
        # 0-based index of the bottom cell.
        kbi=sun.Nkmax-sun.Nk # one based
        nc.variables[vname][:,ii]=kbi

        vname = 'Mesh2_face_top_layer'
        tmp2d = sun.Nkmax-ctop # one based, but inclusive, and >= kbi
        dry=tmp2d<kbi
        tmp2d[dry]=kbi[dry]
        nc.variables[vname][:,ii]=tmp2d

        if False and ii>1:# paranoid testing:
            # array() to drop the mask
            kbi0=np.array(kbi)-1 # 0-based index of deepest FV above the bed.
            kti0=np.array(tmp2d)-1 # 0-based index of upper-most wet FV.
            n_i=nc.dimensions['nMesh2_face'].size
            V_i=nc.variables['Mesh2_face_water_volume'][:,:,ii]
            for i in range(n_i):
                V=V_i[i,:]
                # apparently V is not masked, it just gets 0 below the bed.
                assert (kbi0[i]==0) or np.all( V[:kbi0[i]]==0.0 )
                assert np.all( ~V[kbi0[i]:kti0[i]+1].mask )
                assert np.all( ~V[kbi0[i]:].mask )
                above_surface=V[kti0[i]+1:].data
                assert np.all( np.isnan(above_surface)|(above_surface==0.0) )
        
    print(72*'#')
    print('\t Finished SUNTANS->UnTRIM conversion')
    print(72*'#')

    # close the file
    nc.close()


##############
# Testing
##############

if __name__=="__main__":
    # Inputs
    import argparse

    parser=argparse.ArgumentParser(description='Convert SUNTANS output to UnTRIM/ugrid-ish.')

    parser.add_argument("-i", "--input", help="SUNTANS average output",default="input.nc")
    parser.add_argument("-o", "--output", help="UnTRIM output netcdf", default="output.nc")
    parser.add_argument("-s", "--start", help="Time of start, YYYYMMDD.HHMM",default=None)
    parser.add_argument("-e", "--end", help="Time of start, YYYYMMDD.HHMM",default=None)
    parser.add_argument("-v", "--verbose",help="Increase verbosity",default=1,action='count')

    args=parser.parse_args()

    # ncfile = '../data/Racetrack_AVG_0000.nc'
    # outfile = '../data/Racetrack_untrim.nc'
    # tstart = '20020629.0000'
    # tend = '20060701.1200'

    suntans2untrim(args.input, args.output, args.start, args.end)

