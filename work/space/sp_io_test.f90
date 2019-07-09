program sp_io_test
    use sp_io
    use sp_io_vars
    implicit none
    include 'mpif.h'
    integer :: steps ! total steps
    integer :: tstep ! current timestep (1..steps)
    integer :: i     ! loop
    
    ! local data array
    real*8, dimension(:), allocatable :: wlx, wly, wlz, slx, sly, slz 
    integer*4, dimension(:), allocatable :: wid, sid


    ! initilaize MPI
    call MPI_Init (ierr)
    app_comm = MPI_COMM_WORLD
    call MPI_Comm_rank (app_comm, rank, ierr)
    call MPI_Comm_size (app_comm, nproc, ierr)

    ! determine array size
    lwa = 5
    lwm = 10
    lsa = 15

    twa = nproc * lwa
    twm = nproc * lwm
    tsa = nproc * lsa

    owa = lwa * rank
    owm = lwm * rank
    osa = lsa * rank

    ! allocate and initialize (local) data array
    allocate( wlx(lwa) )
    allocate( wly(lwa) )
    allocate( wlz(lwa) )
    allocate( wid(lwm) )
    allocate( slx(lsa) )
    allocate( sly(lsa) )
    allocate( slz(lsa) )
    allocate( sid(lsa) )

    ! loop
    steps = 10
    do tstep=1, steps
        if (rank == 0) print '("step: ", i4)', tstep
        if (tstep == 1) then
            call io_init()
        endif

        ! For test purpose, reset data here
        do i=1,lwa
            wlx(i) = rank + tstep + i + 10
            wly(i) = rank + tstep + i + 100
            wlz(i) = rank + tstep + i + 1000
        end do
        do i=1, lwm
            wid(i) = rank + tstep + i
        end do
        do i=1, lsa
            slx(i) = rank + tstep + i + 20
            sly(i) = rank + tstep + i + 200
            slz(i) = rank + tstep + i + 2000
            sid(i) = rank + tstep + i + 20000
        end do
        ! end reset data

        call io_write(tstep,wlx,wly,wlz,wid,slx,sly,slz,sid)

        if (tstep == steps) then
            call io_finalize()
        endif
    end do ! steps

    call MPI_Barrier (app_comm, ierr)
    
    ! finalize adios
    ! call io_finalize()
    deallocate (wlx)
    deallocate (wly)
    deallocate (wlz)
    deallocate (wid)
    deallocate (slx)
    deallocate (sly)
    deallocate (slz)
    deallocate (sid)

    ! finalize MPI
    call MPI_Finalize (ierr)
    
end program sp_io_test