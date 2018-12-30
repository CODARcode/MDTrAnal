      module orient_mod
      integer, parameter :: mxelmt = 118
      character*2 :: element(mxelmt)
      contains
c
c     ------------------------------------------------------------------
c
      subroutine init()
      implicit none
      element    = "  "
      element(1) = "H "
      element(6) = "C "
      element(7) = "N "
      element(8) = "O "
      element(15)= "P "
      element(16)= "S "
      element(19)= "K "
      end subroutine init
c
c     ------------------------------------------------------------------
c
      subroutine read_xyz(labels,coords,natom)
      implicit none
      character*2, allocatable, intent(out) :: labels(:)
      double precision, allocatable, intent(out) :: coords(:,:)
      integer, intent(out) :: natom
      integer :: iat
      integer :: ic
      read(*,*,end=200)natom
      read(*,*)
      allocate(labels(1:natom),coords(1:3,1:natom))
      do iat = 1, natom
        read(*,*)labels(iat),(coords(ic,iat),ic=1,3)
        labels(iat)(2:2)=" "
      enddo ! iat
      return
 200  natom=-1
      end subroutine read_xyz
c
c     ------------------------------------------------------------------
c
      integer function atomic_mass(label)
      implicit none
      character*2, intent(in) :: label
      integer :: ielm
      atomic_mass = 0
      do ielm = mxelmt, 1, -1
        if (element(ielm).eq.label) exit
      enddo ! ielm
      atomic_mass = ielm
      if (ielm.eq.0) then
        write(0,*)'Not found: element ',label,'***'
      endif
      return
      end function atomic_mass
c
c     ------------------------------------------------------------------
c
      subroutine calc_center_of_mass(labels,coords,natom,cntr_mass)
      implicit none
      integer, intent(in) :: natom
      character*2, intent(in) :: labels(natom)
      double precision, intent(in) :: coords(3,natom)
      double precision, intent(out) :: cntr_mass(3)
      integer :: ic
      integer :: iat
      double precision :: molwght
      double precision :: atmwght
      molwght = 0.0d0
      do iat = 1, natom
        atmwght = atomic_mass(labels(iat))
        molwght = molwght + atmwght
        do ic = 1, 3
          cntr_mass(ic) = cntr_mass(ic) + atmwght*coords(ic,iat)
        enddo ! ic
      enddo ! iat
      do ic = 1, 3
        cntr_mass(ic) = cntr_mass(ic)/molwght
      enddo ! ic
      end subroutine calc_center_of_mass
c
c     ------------------------------------------------------------------
c
      subroutine recenter_mol(cntr_mass,coords,natom)
      implicit none
      integer, intent(in) :: natom
      double precision, intent(inout) :: coords(3,natom)
      double precision, intent(out) :: cntr_mass(3)
      integer :: ic
      integer :: iat
      do iat = 1, natom
        do ic = 1, 3
          coords(ic,iat) = coords(ic,iat) - cntr_mass(ic)
        enddo ! ic
      enddo ! iat
      end subroutine recenter_mol
c
c     ------------------------------------------------------------------
c
      subroutine calc_moments_inertia(labels,coords,natom,mominert)
      implicit none
      integer, intent(in) :: natom
      character*2, intent(in) :: labels(natom)
      double precision, intent(in) :: coords(3,natom)
      double precision, intent(out) :: mominert(3,3)
      integer :: iat
      double precision :: mass
      double precision :: x
      double precision :: y
      double precision :: z
      mominert = 0.0d0
      do iat = 1, natom
        mass = atomic_mass(labels(iat))
        x = coords(1,iat)
        y = coords(2,iat)
        z = coords(3,iat)
        mominert(1,1)=mominert(1,1)+mass*(y*y+z*z)
        mominert(2,1)=mominert(2,1)-mass* x*y
        mominert(1,2)=mominert(2,1)
        mominert(3,1)=mominert(3,1)-mass* x*z
        mominert(1,3)=mominert(3,1)
        mominert(2,2)=mominert(2,2)+mass*(x*x+z*z)
        mominert(3,2)=mominert(3,2)-mass* y*z
        mominert(2,3)=mominert(3,2)
        mominert(3,3)=mominert(3,3)+mass*(x*x+y*y)
      enddo ! iat
      end subroutine calc_moments_inertia
c
c     ------------------------------------------------------------------
c
      subroutine calc_moments(mominert,vec,eig)
      implicit none
      double precision, intent(in) :: mominert(3,3)
      double precision, intent(out) :: vec(3,3)
      double precision, intent(out) :: eig(3)
      double precision, allocatable :: work(:)
      integer, allocatable :: iwork(:)
      double precision :: t
      integer :: ii, jj, imx
      integer :: lwork, liwork, info
      vec = mominert
      lwork = -1
      liwork = -1
      allocate(work(1:100),iwork(1:100))
      call dsyevd('V','L',3,vec,3,eig,work,lwork,iwork,liwork,info)
      if (info.ne.0) then
        write(0,*)'Work space query for dsyevd failed',info
      endif
      lwork = work(1)+0.5d0
      liwork = iwork(1)
      if (work(1)+0.5d0.ge.100.0d0) then
        deallocate(work)
        allocate(work(1:lwork))
      endif
      if (iwork(1).gt.100) then
        deallocate(iwork)
        allocate(iwork(1:liwork))
      endif
      call dsyevd('V','L',3,vec,3,eig,work,lwork,iwork,liwork,info)
      if (info.ne.0) then
        write(0,*)'Calculation for dsyevd failed',info
      endif
      deallocate(work,iwork)
c
c     fix the sign of the eigenvectors
c
      do jj = 1, 3
        t = 0.0d0
        do ii = 1, 3
          t = t + (4-ii)*vec(ii,jj)
        enddo ! ii
        if (t.lt.0.0d0) then
          do ii = 1, 3
            vec(ii,jj) = -vec(ii,jj)
          enddo ! ii
        endif
      enddo ! jj 
c
c     check right handedness
c
      t = vec(1,3)*( vec(2,1)*vec(3,2) - vec(3,1)*vec(2,2) )
     &  + vec(2,3)*( vec(3,1)*vec(1,2) - vec(1,1)*vec(3,2) )
     &  + vec(3,3)*( vec(1,1)*vec(2,2) - vec(2,1)*vec(1,2) )
      if (t.le.0.0d0) then
        do ii = 1, 3
          vec(ii,3) = -vec(ii,3)
        enddo
      endif
c
c     sort eigenvectors to put maximum component on the diagonal
c
      do ii = 1, 2
        t = abs(vec(ii,ii))
        imx = ii
        do jj = ii+1,3
          if (abs(vec(ii,jj)).gt.t) then
            t = abs(vec(ii,jj))
            imx = jj
          endif
        enddo
        if (imx.ne.ii) then
          do jj=1,3
            t           = vec(jj,ii)
            vec(jj,ii)  = vec(jj,imx)
            vec(jj,imx) = t
          enddo
          t        = eig(ii)
          eig(ii)  = eig(imx)
          eig(imx) = t
        endif
      enddo
c
c     write eigenvector to standard error
c
c     write(0,*)eig
c     write(0,*)
c     do ii=1,3
c       write(0,*)ii,(vec(ii,jj),jj=1,3)
c     enddo
      end subroutine calc_moments
c
c     ------------------------------------------------------------------
c
      subroutine reorient(incrd,vec,outcrd,natom)
      implicit none
      integer :: natom
      double precision, intent(in) :: incrd(3,natom)
      double precision, intent(in) :: vec(3,3)
      double precision, intent(out) :: outcrd(3,natom)
      double precision :: t
      integer :: iat, ic, jc
      do iat = 1, natom
        do jc = 1, 3
          t = 0.0d0
          do ic = 1, 3
            t = t + incrd(ic,iat)*vec(ic,jc)
          enddo ! ic
          outcrd(jc,iat) = t
        enddo ! jc
      enddo ! iat
c
c     Fix the signs once more
c
      do ic = 1, 3
        t = 0.0d0
        do iat = 1, natom
          t = t + (natom-iat+1)*outcrd(ic,iat)
        enddo ! iat
        if (t.lt.0.0d0) then
          do iat = 1, natom
            outcrd(ic,iat) = -outcrd(ic,iat)
          enddo ! iat
        endif
      enddo ! ic
      end subroutine reorient
c
c     ------------------------------------------------------------------
c
      subroutine write_xyz(labels,coords,natom)
      implicit none
      integer, intent(in) :: natom
      character*2, intent(in) :: labels(natom)
      double precision, intent(in) :: coords(3,natom)
      integer :: iat
      integer :: ic
      write(*,*)natom
      write(*,*)
      do iat = 1, natom
        write(*,'(a2,3f24.10)')labels(iat),(coords(ic,iat),ic=1,3)
      enddo ! iat
      end subroutine write_xyz
c
c     ------------------------------------------------------------------
c
      end module orient_mod
c
c-----------------------------------------------------------------------
c
      program orient
      use orient_mod
      character*2, allocatable :: labels(:)
      double precision, allocatable :: incrds(:,:)
      double precision, allocatable :: outcrds(:,:)
      double precision :: cntr_mass(3)
      double precision :: mominert(3,3)
      double precision :: eig(3)
      double precision :: vec(3,3)
      integer :: natom
      natom = 1
      call init
      do while (natom.gt.0)
        call read_xyz(labels,incrds,natom)
        if (natom.le.0) exit
c       write(0,*)'Read ',natom,' atoms'
        call calc_center_of_mass(labels,incrds,natom,cntr_mass)
        call recenter_mol(cntr_mass,incrds,natom)
        call calc_moments_inertia(labels,incrds,natom,mominert)
        call calc_moments(mominert,vec,eig)
        allocate(outcrds(3,natom))
        call reorient(incrds,vec,outcrds,natom)
        call write_xyz(labels,outcrds,natom)
        deallocate(labels,incrds,outcrds)
      enddo
      end program orient
