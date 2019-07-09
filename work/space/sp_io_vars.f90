module sp_io_vars
    ! arguments
    character(len=4095) :: outputfile = 'nwchem_xyz.bp'
    character(len=15)   :: engine_type = 'BPFile' ! engine_type: BPFile (default), SST
    integer :: lwa, twa, owa ! [local, total, offset][water][atoms]
    integer :: lwm, twm, owm ! [local, total, offset][water][molecules]
    integer :: lsa, tsa, osa ! [local, total, offset][solvent][atoms]

    ! MPI 'world' for this app variables
    integer :: app_comm
    integer :: rank, nproc
    integer :: ierr

end module sp_io_vars