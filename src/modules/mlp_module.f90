! Module containing the MLP class to be used in the main script

module mlp_module
    use, intrinsic :: iso_fortran_env, only: real32, int8
    implicit none
    type MLP
        ! Weight matrices
        real(real32), allocatable :: W1(:,:)
        real(real32), allocatable :: W2(:,:)

        ! Output of the forward pass
        real(real32), dimension(:,:), allocatable :: output

        ! Hyperparameters
        integer :: input_size
        integer :: hidden_size
        integer :: output_size
        integer :: batch_size
        real(real32) :: learning_rate
        real(real32) :: loss = 0.0

        ! Methods
        contains
        procedure :: init
        procedure :: forward
        procedure :: backward
        procedure :: train_epoch
        procedure :: predict
        procedure :: get_loss
        procedure :: get_accuracy
    end type MLP

    contains

    function sigmoid(x)
        ! Sigmoid activation function (on batches of vectors)
        real(real32), dimension(:,:), allocatable :: sigmoid
        real(real32), dimension(:,:), intent(in) :: x
        allocate(sigmoid(size(x, 1), size(x, 2)))
        sigmoid = 1.0 / (1.0 + exp(-x))
    end function sigmoid

    function sigmoid_derivative(x)
        ! Derivative of the sigmoid activation function (on batches of vectors)
        real(real32), dimension(:,:), allocatable :: sigmoid_derivative
        real(real32), dimension(:,:), intent(in) :: x
        allocate(sigmoid_derivative(size(x, 1), size(x, 2)))
        sigmoid_derivative = sigmoid(x) * (1.0 - sigmoid(x))
    end function sigmoid_derivative

    function softmax(x)
        ! Softmax activation function (on batches of vectors)
        real(real32), dimension(:,:), allocatable :: softmax
        real(real32), dimension(:,:), intent(in) :: x
        real(real32), dimension(size(x, 1), size(x, 2)) :: exp_x
        real(real32), dimension(size(x, 1)) :: sum_exp_x
        integer :: i

        allocate(softmax(size(x, 1), size(x, 2)))
        exp_x = exp(x)
        sum_exp_x = sum(exp_x, dim=2)
        do i = 1, size(x, 1)
            softmax(i, :) = exp_x(i, :) / sum_exp_x(i)
        end do
    end function softmax

    subroutine init(self, input_size, hidden_size, output_size, batch_size, learning_rate)
        ! Initialize the MLP with random weights between -1 and 1
        class(MLP), intent(inout) :: self
        integer, intent(in) :: input_size, hidden_size, output_size, batch_size
        real(real32), intent(in) :: learning_rate

        self%input_size = input_size
        self%hidden_size = hidden_size
        self%output_size = output_size
        self%batch_size = batch_size
        self%learning_rate = learning_rate


        ! Allocate the weight matrices
        allocate(self%W1(input_size, hidden_size))
        allocate(self%W2(hidden_size, output_size))

        ! Initialize the weights with random values
        call random_number(self%W1)
        call random_number(self%W2)

        ! Scale the weights between -1 and 1
        self%W1 = self%W1 * 2.0 - 1.0
        self%W2 = self%W2 * 2.0 - 1.0

    end subroutine init

    subroutine forward(self, batch, output)
        ! Forward pass of the MLP
        class(MLP), intent(inout) :: self

        real(real32), dimension(:,:), intent(in) :: batch ! The batch is a matrix of size (batch_size, input_size)
        real(real32), dimension(:,:), allocatable, intent(out) :: output ! The output is a matrix of size (batch_size, output_size)

        allocate(output(self%batch_size, self%output_size))

        ! Forward pass (with sigmoid activation function and softmax output layer)
        output = softmax(matmul(sigmoid(matmul(batch, self%W1)), self%W2))
        ! Store the output for the backward pass
        self%output = output

    end subroutine forward

    subroutine backward(self, batch, target)
        ! Backward pass of the MLP
        class (MLP), intent(inout) :: self

        real(real32), dimension(:,:), intent(in) :: batch ! The batch is a matrix of size (batch_size, input_size)
        real(real32), dimension(:,:), intent(in) :: target ! The target is a matrix of size (batch_size, output_size)
        
        real(real32), dimension(self%batch_size, self%output_size) :: delta_t
        real(real32), dimension(self%batch_size, self%hidden_size) :: delta_h

        ! Compute the error of the output layer
        delta_t = (self%output - target ) * sigmoid_derivative(self%output)
        ! Backpropagate the error
        self%W2 = self%W2 - self%learning_rate * matmul(transpose(sigmoid(matmul(batch, self%W1))), delta_t)
        delta_h = matmul(delta_t, transpose(self%W2)) * sigmoid_derivative(matmul(batch, self%W1))
        self%W1 = self%W1 - self%learning_rate * matmul(transpose(batch), delta_h)

    end subroutine backward

    subroutine train_epoch(self, batch, target)
        ! Train the MLP for one epoch on the given batch
        class(MLP), intent(inout) :: self

        real(real32), dimension(:,:), intent(in) :: batch ! The batch is a matrix of size (batch_size, input_size)
        real(real32), dimension(:,:), intent(in) :: target ! The target is a matrix of size (batch_size, output_size)
        ! Forward pass
        call forward(self, batch, self%output)
        ! Backward pass
        call backward(self, batch, target)

    end subroutine train_epoch

    subroutine predict(self, input, prediction)
        ! Predict the class of the input data
        class(MLP), intent(inout) :: self

        real(real32), dimension(:,:), intent(in) :: input
        real, dimension(:), allocatable, intent(out) :: prediction
        real(real32), dimension(:,:), allocatable :: output

        allocate(output(size(input, 1), self%output_size))
        allocate(prediction(size(input, 1)))

        ! Forward pass
        call forward(self, input, output)

        ! Predict the class with the highest probability
        prediction = maxloc(output, dim=2) - 1

    end subroutine predict

    subroutine get_loss(self, target, loss)
        ! Compute the loss of the MLP
        class(MLP), intent(inout) :: self

        real(real32), dimension(:,:), intent(in) :: target ! The target is a matrix of size (batch_size, output_size)
        real(real32), intent(out) :: loss

        ! Compute the cross-entropy loss
        loss = loss -sum(target * log(self%output))

    end subroutine get_loss

    subroutine get_accuracy(self, input, target, accuracy)
        ! Compute the accuracy of the MLP
        class(MLP), intent(inout) :: self

        real (real32), dimension(:,:), intent(in) :: input 
        real(real32), dimension(:), intent(in) :: target
        real(real32), intent(out) :: accuracy

        real, dimension(:), allocatable :: prediction
        real(real32) :: correct_predictions

        ! Predict the classes
        call predict(self, input, prediction)

        ! Compute the accuracy
        correct_predictions = sum(merge(1.0, 0.0, prediction == target))
        accuracy = correct_predictions / size(target)

    end subroutine get_accuracy

end module mlp_module

