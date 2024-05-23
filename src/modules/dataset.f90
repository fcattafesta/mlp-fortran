! Module containing the routines to load and preprocess the data

module dataset

    use iso_fortran_env, only: real32, int8
    implicit none

    contains
    subroutine load_data(X, y, filename)
    
        ! Inputs
        character(len=*), intent(in) :: filename
        ! Outputs
        real(real32), dimension(:,:), allocatable, intent(out) :: X
        real(real32), dimension(:), allocatable, intent(out) :: y
        ! Local
        integer :: i, ios, iu, n_samples
        logical :: exists
    
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
    
    subroutine normalize(X)
        
        real(real32), dimension(:,:), intent(inout) :: X
        real(real32) :: max_val
    
        max_val = maxval(X)
        X = X / max_val
    
    end subroutine normalize
    
    subroutine one_hot_encode(y, y_encoded)
    
        real(real32), dimension(:), intent(in) :: y
        real(real32), dimension(size(y), 10), intent(inout) :: y_encoded
        integer :: i
        ! Initialize the encoded array to zero everywhere except for the diagonal elements which are set to 1
        y_encoded = 0
        do i = 1, size(y)
            y_encoded(i, int(y(i)) + 1) = 1
        end do
    
    end subroutine one_hot_encode
    
end module dataset