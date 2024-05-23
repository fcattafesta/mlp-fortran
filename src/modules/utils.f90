module utils

    use iso_fortran_env, only: real32
    implicit none
    
contains
    subroutine print_picture(X, label)
        ! Print a picture and its label to the console
        real(real32), dimension(:), intent(in) :: X
        real(real32), dimension(28, 28) :: picture
        real(real32), dimension(:), intent(in) :: label
        integer :: i, j

        ! Print a picture and its label
        picture = transpose(reshape(X, [28, 28]))
        print "(a)", "Example picture:"
        do i = 1, 28
            do j = 1, 28
                if (picture(i, j) > 0.1) then
                    write(*, '(a)', advance='no') 'x'
                else
                    write(*, '(a)', advance='no') ' '
                end if
            end do
            print*
        end do
        write(*, '(a)', advance='no') "Label: [ "
        do i = 1, size(label)
            if (i < size(label)) then
                write(*, '(f0.0, a)', advance='no') label(i), ", "
            else
                write(*, '(f0.0)', advance='no') label(i)
            end if
        end do
        print *, "]" 
    end subroutine print_picture
    

    subroutine write_to_file(epochs, train_loss, test_loss, train_accuracy, test_accuracy, filename)
        ! Write the metrics to a csv file
        integer, intent(in) :: epochs
        real(real32), dimension(:), intent(in) :: train_loss, test_loss, train_accuracy, test_accuracy
        character(len=*), intent(in) :: filename
        integer :: i, iu, ios

        ! Open the file
        open(unit=iu, file=filename, action="write", iostat=ios)
        if (ios /= 0) then
            print *, "Error opening file"
            stop
        end if

        ! Write the header
        write(iu, '(a)') "epoch, train_loss, test_loss, train_accuracy, test_accuracy"

        ! Write the metrics
        do i = 1, epochs
            write(iu, '(i0, 4(",", f0.4))') i, train_loss(i), test_loss(i), train_accuracy(i), test_accuracy(i)
        end do
        ! Close the file
        close(iu)

    end subroutine write_to_file

    subroutine argument_parser(batch_size, epochs, hidden_size, learning_rate, train_file, test_file, metrics_file)

        implicit none
        character(len=100) :: arg
        integer :: i, n_args

        integer, intent(out) :: batch_size, epochs, hidden_size
        real(real32), intent(out) :: learning_rate
        character(len=100), intent(out) :: train_file, test_file, metrics_file

        ! Get the number of arguments
        n_args = command_argument_count()
        ! No positional arguments, only keyword arguments. If the keyword is not found, the default value is used
        batch_size = 32
        epochs = 10
        hidden_size = 128
        learning_rate = 0.01
        train_file = "data/mnist_train.csv"
        test_file = "data/mnist_test.csv"
        metrics_file = "data/metrics.csv"

        ! Print usage if the argument -h or --help is passed
        do i = 1, n_args
            call get_command_argument(i, arg)
            if (arg == "-h" .or. arg == "--help") then
                print *, "usage: ./train [option]"
                print *, "Options:"
                print *, "  --batch_size <int>    Batch size for training (default: 32)"
                print *, "  --epochs <int>        Number of epochs (default: 10)"
                print *, "  --hidden_size <int>   Number of hidden units (default: 128)"
                print *, "  --learning_rate <float> Learning rate (default: 0.01)"
                print *, "  --train_file <string> Path to the training file (default: data/mnist_train.csv)"
                print *, "  --test_file <string>  Path to the test file (default: data/mnist_test.csv)"
                print *, "  --metrics_file <string> Path to the metrics file (default: data/metrics.csv)"
                stop
            end if
        end do

        ! Loop through the arguments
        do i = 1, n_args
            ! Get the argument
            call get_command_argument(i, arg)
            ! Parse the argument
            select case (arg)
                case ("--batch_size")
                    call get_command_argument(i + 1, arg)
                    read(arg, *) batch_size
                case ("--epochs")
                    call get_command_argument(i + 1, arg)
                    read(arg, *) epochs
                case ("--hidden_size")
                    call get_command_argument(i + 1, arg)
                    read(arg, *) hidden_size
                case ("--learning_rate")
                    call get_command_argument(i + 1, arg)
                    read(arg, *) learning_rate
                case ("--train_file")
                    call get_command_argument(i + 1, arg)
                    train_file = trim(arg)
                case ("--test_file")
                    call get_command_argument(i + 1, arg)
                    test_file = trim(arg)
                case ("--metrics_file")
                    call get_command_argument(i + 1, arg)
                    metrics_file = trim(arg)
            end select
        end do

        ! Print the arguments
        print *, "Options:"
        print *, "  Batch size: ", batch_size
        print *, "  Epochs: ", epochs
        print *, "  Hidden size: ", hidden_size
        print *, "  Learning rate: ", learning_rate
        print *, "  Train file: ", train_file
        print *, "  Test file: ", test_file
        print *, "  Metrics file: ", metrics_file
    
    end subroutine argument_parser

end module utils