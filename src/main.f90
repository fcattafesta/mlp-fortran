! Main program containing also the command line parser
subroutine load_data(X, y, filename)

    use iso_fortran_env, only: real32, int8

    implicit none
    ! Inputs
    character(len=*), intent(in) :: filename
    ! Outputs
    real(real32), dimension(:,:), allocatable, intent(out) :: X
    integer(int8), dimension(:), allocatable, intent(out) :: y
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

    use iso_fortran_env, only: real32

    implicit none
    real(real32), dimension(:,:), intent(inout) :: X
    real(real32) :: max_val

    max_val = maxval(X)
    X = X / max_val

end subroutine normalize

subroutine one_hot_encode(y, y_encoded)

    use iso_fortran_env, only: int8

    implicit none
    integer(int8), dimension(:), intent(in) :: y
    integer(int8), dimension(size(y), 10), intent(inout) :: y_encoded
    integer :: i
    ! Initialize the encoded array to zero everywhere except for the diagonal elements which are set to 1
    y_encoded = 0
    do i = 1, size(y)
        y_encoded(i, y(i) + 1) = 1
    end do

end subroutine one_hot_encode


program main

    use iso_fortran_env, only: real32, int8

    implicit none
    real(real32), dimension(:,:), allocatable :: X_train, X_test, picture
    integer(int8), dimension(:), allocatable :: y_train, y_test
    integer(int8), dimension(:,:), allocatable :: y_train_encoded, y_test_encoded
    ! character(len=256) :: filename_train, filename_test
    integer :: i, j

    interface
        subroutine load_data(X, y, filename)
            use iso_fortran_env, only: real32, int8
            implicit none
            real(real32), dimension(:,:), allocatable, intent(out) :: X
            integer(int8), dimension(:), allocatable, intent(out) :: y
            character(len=*), intent(in) :: filename
        end subroutine load_data
    end interface

    interface
        subroutine normalize(X)
            use iso_fortran_env, only: real32
            implicit none
            real(real32), dimension(:,:), intent(inout) :: X
        end subroutine normalize
    end interface

    interface
        subroutine one_hot_encode(y, y_encoded)
            use iso_fortran_env, only: int8
            implicit none
            integer(int8), dimension(:), intent(in) :: y
            integer(int8), dimension(size(y), 10), intent(inout) :: y_encoded
        end subroutine one_hot_encode
    end interface
    
    call load_data(X_train, y_train, 'data/mnist_train.csv')
    call load_data(X_test, y_test, 'data/mnist_test.csv')

    print*, 'X_train shape: ', shape(X_train)
    print*, 'y_train shape: ', shape(y_train)
    print*, 'X_test shape: ', shape(X_test)
    print*, 'y_test shape: ', shape(y_test)

    call normalize(X_train)
    call normalize(X_test)

    allocate(y_train_encoded(size(y_train), 10))
    allocate(y_test_encoded(size(y_test), 10))

    call one_hot_encode(y_train, y_train_encoded)
    call one_hot_encode(y_test, y_test_encoded)

    print*, 'y_train_encoded shape: ', shape(y_train_encoded)

    picture = transpose(reshape(X_train(3, :), [28, 28]))
    

    ! nicely print the picture (rows are columns in Fortran)
    do i = 1, 28
        do j = 1, 28
            if (picture(i, j) > 0.1) then
                write(*, '(A)', advance='no') 'x'
            else
                write(*, '(A)', advance='no') ' '
            end if
        end do
        print*
    end do

    ! print*, 'y_train: ', y_train_encoded(3, :)



end program main