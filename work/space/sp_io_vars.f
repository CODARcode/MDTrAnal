      module sp_io_vars
        ! output file name
        character(len=4095) :: outputfile = 'nwchem_xyz.bp'
        ! engine_type: BPFile (default), SST
        character(len=15)   :: engine_type = 'BPFile'
        ! [local, total, offset][water][atoms] 
        integer :: lwa, twa, owa 
        ! [local, total, offset][water][molecules]
        integer :: lwm, twm, owm 
        ! [local, total, offset][solvent][atoms]
        integer :: lsa, tsa, osa 

        ! MPI 'world' for this app variables
        integer :: app_comm
        integer :: rank, nproc
        integer :: ierr
      end module sp_io_vars