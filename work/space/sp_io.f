      module sp_io
        use adios2
        implicit none
        ! adios handlers
        type(adios2_adios)    :: adios      
        type(adios2_io)       :: io         
        type(adios2_engine)   :: bp_writer
    
        type(adios2_variable) :: varWLX, varWLY, varWLZ, varWID
        type(adios2_variable) :: varSLX, varSLY, varSLZ, varSID

      contains


      subroutine io_init()
        use sp_io_vars
        use adios2
        implicit none

        call adios2_init (adios, app_comm, ierr)
        call adios2_declare_io (io, adios, 'coordinates', ierr)
        call adios2_set_engine (io, engine_type, ierr)
      end subroutine io_init


      subroutine io_finalize()
        use sp_io_vars
        use adios2
        implicit none

        call adios2_close (bp_writer, ierr)
        call adios2_finalize (adios, ierr)
      end subroutine io_finalize


      subroutine io_write(tstep, wlx, wly, wlz, wid, slx, sly, slz, sid)
        use sp_io_vars
        use adios2
        implicit none

        integer, intent(in)   :: tstep
        real*8, intent(in)    :: wlx(:), wly(:), wlz(:)
        real*8, intent(in)    :: slx(:), sly(:), slz(:)
        integer*4, intent(in) :: wid(:), sid(:)

        integer*4 :: istatus
        integer*8, dimension(1) :: shape_dims
        integer*8, dimension(1) :: start_dims
        integer*8, dimension(1) :: count_dims
        integer*4 :: ndims=1

        ! Define variables at the first time
        print *, 'io_write: ', tstep
        if (tstep.eq.1) then
          call adios2_open(bp_writer, io, outputfile, 
     &       adios2_mode_write, ierr)
          if (ierr.ne.0) then
            print *,"Failed to open bp file!"
          endif
        
          ! Define array dimensions
          ! WATER
          shape_dims(1) = twa
          start_dims(1) = owa
          count_dims(1) = lwa
          call adios2_define_variable (varWLX, io, "WLX", 
     &      adios2_type_dp, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)
          call adios2_define_variable (varWLY, io, "WLY", 
     &      adios2_type_dp, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)
          call adios2_define_variable (varWLZ, io, "WLZ", 
     &      adios2_type_dp, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)

          shape_dims(1) = twm
          start_dims(1) = owm
          count_dims(1) = lwm
          call adios2_define_variable (varWID, io, "WID", 
     &      adios2_type_integer4, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)

          ! SOLVENT                    
          shape_dims(1) = tsa
          start_dims(1) = osa
          count_dims(1) = lsa
          call adios2_define_variable (varSLX, io, "SLX", 
     &      adios2_type_dp, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)
          call adios2_define_variable (varSLY, io, "SLY", 
     &      adios2_type_dp, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)
          call adios2_define_variable (varSLZ, io, "SLZ", 
     &      adios2_type_dp, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)
          call adios2_define_variable (varSID, io, "SID", 
     &      adios2_type_integer4, ndims, 
     &      shape_dims, start_dims, count_dims, 
     &      adios2_constant_dims, ierr)
        endif

        ! barrier needed???
        call MPI_BARRIER(app_comm, ierr)

        ! block untile step available (Timeout = -1.)
        call adios2_begin_step(bp_writer, 
     &    adios2_step_mode_append, -1., istatus, ierr)

        call adios2_put(bp_writer, varWLX, wlx, ierr)
        call adios2_put(bp_writer, varWLY, wly, ierr)
        call adios2_put(bp_writer, varWLZ, wlz, ierr)
        call adios2_put(bp_writer, varWID, wid, ierr)
        call adios2_put(bp_writer, varSLX, slx, ierr)
        call adios2_put(bp_writer, varSLY, sly, ierr)
        call adios2_put(bp_writer, varSLZ, slz, ierr)
        call adios2_put(bp_writer, varSID, sid, ierr)

        call adios2_end_step(bp_writer, ierr)
      end subroutine io_write

      end module sp_io
