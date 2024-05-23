! Main program containing also the command line parser

program main

    use iso_fortran_env, only: real32, int8
    use dataset
    use mlp_module, only: MLP
    use utils

    implicit none
    ! Train and test data
    real(real32), dimension(:,:), allocatable :: X_train, X_test, picture
    real(real32), dimension(:), allocatable :: y_train, y_test
    real(real32), dimension(:,:), allocatable :: y_train_encoded, y_test_encoded
    ! Neural network
    type(MLP) :: nn
    integer :: batch_size, epochs, hidden_size, input_size, output_size
    real(real32) :: learning_rate, random_index
    ! Metrics
    real(real32), dimension(:), allocatable :: train_loss, test_loss, train_accuracy, test_accuracy, epochs_vector
    ! Others
    integer :: i, j, epoch
    character(len=100) :: train_file, test_file, metrics_file

    ! Parse the command line arguments
    call argument_parser(batch_size, epochs, hidden_size, learning_rate, train_file, test_file, metrics_file)

    ! Allocate the metrics arrays
    allocate(epochs_vector(epochs))
    allocate(train_loss(epochs))
    allocate(test_loss(epochs))
    allocate(train_accuracy(epochs))
    allocate(test_accuracy(epochs))

    ! Load the data !! Pass the file names as arguments
    call load_data(X_train, y_train, train_file)
    call load_data(X_test, y_test, test_file)

    ! Print the shapes of the data in a nice way
    print "(a)", "Data loaded:"
    write(*, '(a, 2(i0, a))') "X_train shape: (", size(X_train, dim=1), ", ", size(X_train, dim=2), ")"
    write(*, '(a, 2(i0, a))') "y_train shape: (", size(y_train), ")"
    write(*, '(a, 2(i0, a))') "X_test shape: (", size(X_test, dim=1), ", ", size(X_test, dim=2), ")"
    write(*, '(a, 2(i0, a))') "y_test shape: (", size(y_test), ")"

    ! Normalize the data
    call normalize(X_train)
    call normalize(X_test)

    ! Allocate and One-hot encode the labels
    allocate(y_train_encoded(size(y_train), 10))
    allocate(y_test_encoded(size(y_test), 10))
    call one_hot_encode(y_train, y_train_encoded)
    call one_hot_encode(y_test, y_test_encoded)

    call random_number(random_index)

    random_index = random_index * size(X_train, 1)

    call print_picture(X_train(int(random_index), :), y_train_encoded(int(random_index), :))

    ! Initialize the neural network
    input_size = size(X_train, 2)
    output_size = size(y_test_encoded, 2)
    call nn%init(input_size, hidden_size, output_size, batch_size, learning_rate)

    
    ! Train the neural network
    do epoch=1, epochs
        ! Train Loop
        ! Loop over the batches
        do j=1, size(X_train, 1), batch_size
            call nn%train_epoch(X_train(j:j+batch_size-1, :), y_train_encoded(j:j+batch_size-1, :))
            call nn%get_loss(y_train_encoded(j:j+batch_size-1, :), train_loss(epoch))
        end do
        ! Compute the train loss and accuracy
        train_loss(epoch) = train_loss(epoch) / size(X_train, 1)
        call nn%get_accuracy(X_train, y_train, train_accuracy(epoch))
        ! Test Loop
        ! Loop over the batches
        do j=1, size(X_test, 1), batch_size
            call nn%forward(X_test(j:j+batch_size-1, :), nn%output)
            call nn%get_loss(y_test_encoded(j:j+batch_size-1, :), test_loss(epoch))
        end do  
        ! Compute the test loss and accuracy
        test_loss(epoch) = test_loss(epoch) / size(X_test, 1)
        call nn%get_accuracy(X_test, y_test, test_accuracy(epoch))
        ! Print the metrics
        write(*, '(a, i3)', advance='no') "Epoch: ", epoch
        write(*, '(a, f6.4)', advance='no') " - Train Loss: ", train_loss(epoch)
        write(*, '(a, f6.4)', advance='no') " - Test Loss: ", test_loss(epoch)
        write(*, '(a, f6.4)', advance='no') " - Train Accuracy: ", train_accuracy(epoch)
        write(*, '(a, f6.4)', advance='yes') " - Test Accuracy: ", test_accuracy(epoch)
    end do

    call write_to_file(epochs, train_loss, test_loss, train_accuracy, test_accuracy, metrics_file)

end program main