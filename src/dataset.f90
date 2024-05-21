! Module containing the routines to load and preprocess the data

module dataset

    use iso_fortran_env, only: real32, int8

    contains
    subroutine load_data(X, y, filename)

        implicit none
        ! Inputs
        character(len=*), intent(in) :: filename
        ! Outputs
        real(real32), dimension(:,:), allocatable, intent(out) :: X
        integer(int8), dimension(:), allocatable, intent(out) :: y
        ! Local
        integer :: i, ios, iu, exists, n_samples

        ! Open the file
        inquire(file=filename, exist=exists)
        if (.not.exists) then
            print*, 'Error: file ', trim(filename), ' does not exist'
            stop
        end if
        open(newunit=iu, file=filename, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            print*, 'Error: could not open file ', trim(filename)
            stop
        end if

        ! Get the length of the file first (number of samples)
        n_samples = 0
        do
            read(iu, *, iostat=ios)
            if (ios /= 0) exit
            n_samples = n_samples + 1
        end do
        rewind(iu)

        ! Allocate the arrays (we are assuming that the horizontal dimension is fixed to 784 + 1)
        allocate(X(n_samples, 784), y(n_samples))

        ! Read the data
        ! The first column is the true label and the rest are the pixel values
        do i = 1, n_samples
            read(iu, *) y(i), X(i, :)
        end do

        ! Close the file
        close(iu)

    end subroutine load_data






    
end module dataset