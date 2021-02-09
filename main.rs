extern crate rand;
use rand::prelude::*;
use relative_path::RelativePath;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant};

mod matrix;
use crate::matrix::Matrix;

fn sigmoid(input: f32) -> f32 {
    // Returns sigmoid of the input
    1.0 / (1.0 + (-input).exp())
}

fn relu(input: f32) -> f32 {
    // Returns relu of the input
    0.0_f32.max(input)
}

fn relu_prime(input: f32) -> f32 {
    // Return the derivative of relu of input
    if input >= 0.0 {
        1.0
    } else {
        0.0
    }
}

fn get_acc(predictions: &Matrix, targets: &Matrix) -> f32 {
    /// Compute accuracy
    /// # Arguments
    /// param:predictions - matrix of predictions, only 1 row
    /// param:targets - matrix of expected values, only 1 row
    let num_of_samples = targets.columns as f32;
    let mut error = 0.0;
    for sample in 0..targets.columns {
        if (targets.data[0][sample] - predictions.data[0][sample]) == 0.0 {
            error = error + 1.0;
        }
    }
    error / num_of_samples
}

fn generate_train_validation_split(total: usize, validation_size: usize) -> Vec<usize> {
    /// Method for generating split of training input data into train and validation parts
    /// # Arguments
    /// param:total - total number of training samples
    /// param:validation_size - number of samples that should be in validation set
    // init part - first part of input data is training part, the end is validation part
    let mut vec = Vec::with_capacity(validation_size);
    for i in (total - validation_size)..total {
        vec.push(i);
    }

    let mut rng = rand::thread_rng();
    let mut ind = 0;
    let prob = (validation_size as f32) / (total as f32);
    for number in 0..total {
        if rng.gen::<f32>() < prob {
            // with probability prob the sample would be chosen as validation
            vec[ind] = number;
            ind += 1;
        }
        if total - number <= validation_size - ind || ind >= validation_size {
            // if we have sampled the required number of validation data (indices) OR
            // we have to use all remaining samples as validation
            // => we can stop
            break;
        }
    }
    vec
}

fn read_csv(path: &str, n_rows: usize, n_cols: usize) -> Matrix {
    /// Method for reading input csv file into a matrix, the matrix is transposed, rows <-> columns
    /// # Arguments
    /// param:path - Path to the csv file that should be read
    /// param:n_rows - The number of rows in csv, the number of columns in the output Matrix
    /// param:n_cols - The number of columns in csv, the number of rows in the output Matrix
    let mut output_tr = Matrix::new(n_cols, n_rows);

    let f = BufReader::new(File::open(path).expect("open failed"));
    let mut col = 0;
    for line in f.lines() {
        let mut row = 0;
        for c in line.expect("lines failed").chars() {
            match c {
                ',' => {
                    output_tr.data[row][col] = output_tr.data[row][col] / 255.0;
                    row += 1
                }
                _ => {
                    output_tr.data[row][col] =
                        output_tr.data[row][col] * 10.0 + (c as i32 - 48) as f32
                }
            }
        }
        col += 1;
    }
    output_tr
}

fn read_csv_with_split(
    path: &str,
    n_rows: usize,
    n_cols: usize,
    split: &Vec<usize>,
) -> (Matrix, Matrix) {
    /// Method for reading input csv file into variable with data for training and variable with data for validation based on a split vector
    /// # Arguments
    /// param:path - Path to the csv file that should be read
    /// param:n_rows - The number of columns in csv, the number of rows in the output matrices
    /// param:n_cols - The number of rows in csv, equals sum of the number of columns in the output matrices
    /// param:split - Vector of indices - samples from input csv that will be loaded into validation set
    let validation_size = split.len();
    let mut train = Matrix::new(n_rows, n_cols - validation_size);
    let mut validate = Matrix::new(n_rows, validation_size);
    let f = BufReader::new(File::open(path).expect("open failed"));

    let mut ind = 0;
    let mut col = 0;

    for line in f.lines() {
        if ind < validation_size && col == split[ind] {
            // belongs to validation set
            let mut row = 0;
            for c in line.expect("lines failed").chars() {
                match c {
                    ',' => {
                        validate.data[row][ind] = validate.data[row][ind] / 255.0;
                        row += 1
                    }
                    _ => {
                        validate.data[row][ind] =
                            validate.data[row][ind] * 10.0 + (c as i32 - 48) as f32
                    }
                }
            }
            if ind < validation_size {
                ind += 1;
            }
        } else {
            // belongs to training set
            let mut row = 0;
            for c in line.expect("lines failed").chars() {
                match c {
                    ',' => {
                        train.data[row][col - ind] = train.data[row][col - ind] / 255.0;
                        row += 1
                    }
                    _ => {
                        train.data[row][col - ind] =
                            train.data[row][col - ind] * 10.0 + (c as i32 - 48) as f32
                    }
                }
            }
        }
        col += 1;
    }
    (train, validate)
}

fn write_csv(path: &str, m: &Matrix) {
    /// Method for writing matrix data into csv file
    /// # Arguments
    /// param:path - Path to the csv file that should be written into
    /// param:m - The Matrix that hsould be written
    let f = File::create(path).unwrap();
    let mut wrt = BufWriter::new(f);
    let mut num = 2;

    for colrow in 0..m.columns {
        while num >= 0 {
            let ch = (m.data[0][colrow] / (10.0_f32.powi(num))) as i32;
            if ch > 0 || (num == 2 && m.data[0][colrow] == 0.0) {
                let ch = ((ch as u32 % 10) as u8) + '0' as u8;
                wrt.write(&[ch]);
            }
            num = num - 1;
        }
        wrt.write(&['\n' as u8]);
        num = 2;
    }
}

fn generate_indices_for_batch(batch_size: usize, max_number: usize) -> Matrix {
    // The function generates indices of rows/columns that should be part of a batch
    // In each line of output are indices for one batch.
    // param:batch_size - size of one batch
    // param:max_number - maximal value of generated indices (number of columns/rows)

    let mut rng = rand::thread_rng();
    let mut indices = Matrix::new(1, batch_size);
    for c in 0..batch_size {
        indices.data[0][c] = rng.gen_range(0, max_number) as f32;
    }
    indices
}

#[derive(Debug)]
struct Network {
    /// Network for multi class classification, output layer uses softmax
    /// All data matrices are assumed to be of the form features x samples
    /// # Arguments
    /// param:input_dim - takes the dimensionality of input features, in this project always 28*28
    /// param:batch_size - takes the number of samples in batch dataset
    /// param:val_size - takes the number of samples in validation dataset
    /// param:sizes - takes a vector containing the number of neurons in each layer, dimensions: 1xlayers
    /// param:W - vector of matrices of weights between the previous and current layer, dimensions: 1xlayers
    /// param:b - vector of single column matrices containing bias terms for each layer, dimensions: 1xlayers
    /// param:batch_activations - vector of matrices of batch activations for given layer, dimensions: 1x(layers+1)
    /// param:batch_potentials - vector of matrices of batch potentials for given layer, dimensions: 1x(layers+1)
    /// param:val_activations - vector of matrices of val activations for given layer, dimensions: 1x(layers+1)
    /// param:val_potentials - vector of matrices of val potentials for given layer, dimensions: 1x(layers+1)
    /// param:test_activations - vector of matrices of val activations for given layer, dimensions: 1x(layers+1)
    /// param:test_potentials - vector of matrices of val potentials for given layer, dimensions: 1x(layers+1)
    /// param:derivative_W - vector of matrices of weight derivatives, dimensions: 1x(layers)
    /// param:derivative_b - vector of single column matrices containing bias derivatives, dimensions: 1x(layers)
    /// param:deltas - vector of matrices of deltas, dimensions: 1x(layers)
    /// param:velocities_W - vector of matrices of velocities for weights, dimensions: 1xlayers
    /// param:velocities_b - vector of matrices of velocities for biases, dimensions: 1xlayers
    /// param:summation_W - vector of matrices of weight summations from RMSprob, dimensions: 1xlayers
    /// param:summation_b - vector of matrices of bias summations from RMSprob, dimensions: 1xlayers
    /// param:backprop_m_holder - instantiates vector of matrices holders for temporary calculations in backpropagation step, dimensions: 1xlayers
    /// param:backprop_z_holder - instantiates vector of matrices holders for temporary calculations in backpropagation step, dimensions: 1xlayers
    input_dim: usize,
    batch_size: usize,
    val_size: usize,
    sizes: Vec<usize>,
    layers: usize,
    W: Vec<Matrix>,
    b: Vec<Matrix>,
    batch_activations: Vec<Matrix>,
    batch_potentials: Vec<Matrix>,
    val_activations: Vec<Matrix>,
    val_potentials: Vec<Matrix>,
    test_activations: Vec<Matrix>,
    test_potentials: Vec<Matrix>,
    derivative_W: Vec<Matrix>,
    derivative_b: Vec<Matrix>,
    deltas: Vec<Matrix>,
    velocities_W: Vec<Matrix>,
    velocities_b: Vec<Matrix>,
    summation_W: Vec<Matrix>,
    summation_b: Vec<Matrix>,
    backprop_m_holder: Vec<Matrix>,
    backprop_z_holder: Vec<Matrix>,
}

impl Network {
    fn new(input_dim: usize, sizes: Vec<usize>, batch_size: usize, val_size: usize) -> Network {
        /// Constructor for Network struct
        /// # Arguments
        /// param:input_dim - takes the dimensionality of input features, in this project always 28*28
        /// param:sizes - takes a vector containing the number of neurons in each layer, dimensions: 1xlayers
        /// param:batch_size - takes the number of samples in batch dataset
        /// param:val_size - takes the number of samples in validation dataset
        let mut W = Vec::new();
        let mut b = Vec::new();
        let mut batch_activations = Vec::new();
        let mut batch_potentials = Vec::new();
        let mut val_activations = Vec::new();
        let mut val_potentials = Vec::new();
        let mut test_activations = Vec::new();
        let mut test_potentials = Vec::new();
        let mut derivative_W = Vec::new();
        let mut derivative_b = Vec::new();
        let mut deltas = Vec::new();
        let mut velocities_W = Vec::new();
        let mut velocities_b = Vec::new();
        let mut summation_W = Vec::new();
        let mut summation_b = Vec::new();
        let mut backprop_m_holder = Vec::new();
        let mut backprop_z_holder = Vec::new();

        let mut previous_layer_size = input_dim;
        let mut layer_idx = 0;

        batch_activations.push(Matrix::new(input_dim, batch_size));
        batch_potentials.push(Matrix::new(input_dim, batch_size));
        val_activations.push(Matrix::new(input_dim, val_size));
        val_potentials.push(Matrix::new(input_dim, val_size));

        for layer_size in sizes.iter() {
            // # normalized intitiliazation
            let upper = (2.0 / (previous_layer_size.clone() as f32)).sqrt();

            W.push(Matrix::randn_new(
                layer_size.clone(),
                previous_layer_size.clone(),
                0.0,
                upper,
            ));
            b.push(Matrix::new(layer_size.clone(), 1));
            batch_activations.push(Matrix::new(layer_size.clone(), batch_size));
            batch_potentials.push(Matrix::new(layer_size.clone(), batch_size));
            val_activations.push(Matrix::new(layer_size.clone(), val_size));
            val_potentials.push(Matrix::new(layer_size.clone(), val_size));
            derivative_W.push(Matrix::new(layer_size.clone(), previous_layer_size.clone()));
            derivative_b.push(Matrix::new(layer_size.clone(), 1));
            deltas.push(Matrix::new(layer_size.clone(), batch_size));
            velocities_W.push(Matrix::new(layer_size.clone(), previous_layer_size.clone()));
            velocities_b.push(Matrix::new(layer_size.clone(), 1));
            summation_W.push(Matrix::new(layer_size.clone(), previous_layer_size.clone()));
            summation_b.push(Matrix::new(layer_size.clone(), 1));

            if layer_idx < sizes.len() - 1 {
                backprop_m_holder.push(Matrix::new(
                    deltas[layer_idx].rows,
                    deltas[layer_idx].columns,
                ));
                backprop_z_holder.push(Matrix::new(
                    batch_potentials[layer_idx + 1].rows,
                    batch_potentials[layer_idx + 1].columns,
                ));
            }

            previous_layer_size = layer_size.clone();
            layer_idx += 1;
        }

        Network {
            input_dim: input_dim,
            batch_size: batch_size,
            val_size: val_size,
            layers: sizes.len(),
            sizes: sizes,
            W: W,
            b: b,
            batch_activations: batch_activations,
            batch_potentials: batch_potentials,
            val_activations: val_activations,
            val_potentials: val_potentials,
            test_activations: test_activations,
            test_potentials: test_potentials,
            derivative_W: derivative_W,
            derivative_b: derivative_b,
            deltas: deltas,
            velocities_W: velocities_W,
            velocities_b: velocities_b,
            summation_W: summation_W,
            summation_b: summation_b,
            backprop_m_holder: backprop_m_holder,
            backprop_z_holder: backprop_z_holder,
        }
    }

    fn feed_forward(&mut self, data: &Matrix, activation_fn: &dyn Fn(f32) -> f32, keep_prob: f32) {
        /// Feed forward method
        /// # Arguments
        /// param:data - Matrix of input data features x samples
        /// param:activation_fn - activation function which will be applied to all hidden layers
        // # training branch
        if data.columns == self.batch_size {
            self.batch_activations[0] = data.clone();
            for layer in 0..self.layers {
                self.batch_potentials[layer + 1]
                    .multAB(&self.W[layer], &self.batch_activations[layer]);
                self.batch_potentials[layer + 1].c_add_vector(&self.b[layer]);

                if layer == self.layers - 1 {
                    self.batch_activations[layer + 1]
                        .c_apply_softmax(&self.batch_potentials[layer + 1]);
                } else {
                    matrix::e_apply_function(
                        &mut self.batch_activations[layer + 1],
                        &self.batch_potentials[layer + 1],
                        &activation_fn,
                    );
                }
            }
        }
        // # val prediction branch
        else if data.columns == self.val_size {
            self.val_activations[0] = data.clone();
            for layer in 0..self.layers {
                self.val_potentials[layer + 1].multAB(&self.W[layer], &self.val_activations[layer]);
                self.val_potentials[layer + 1].c_add_vector(&self.b[layer]);

                if layer == self.layers - 1 {
                    self.val_activations[layer + 1]
                        .c_apply_softmax(&self.val_potentials[layer + 1]);
                } else {
                    matrix::e_apply_function(
                        &mut self.val_activations[layer + 1],
                        &self.val_potentials[layer + 1],
                        &activation_fn,
                    );
                }
            }
        }
        // # test prediction branch
        else {
            self.initialize_test_matrices(data.columns);
            self.test_activations[0] = data.clone();
            for layer in 0..self.layers {
                self.test_potentials[layer + 1]
                    .multAB(&self.W[layer], &self.test_activations[layer]);
                self.test_potentials[layer + 1].c_add_vector(&self.b[layer]);

                if layer == self.layers - 1 {
                    self.test_activations[layer + 1]
                        .c_apply_softmax(&self.test_potentials[layer + 1]);
                } else {
                    matrix::e_apply_function(
                        &mut self.test_activations[layer + 1],
                        &self.test_potentials[layer + 1],
                        &activation_fn,
                    );
                }
            }
        }
    }

    fn back_propagate(
        &mut self,
        targets: &Matrix,
        hidden_activation_fn_prime: &dyn Fn(f32) -> f32,
    ) {
        /// Back propagate method
        /// # Arguments
        /// param:target - Matrix of targets 1 x samples
        /// param:hidden_activation_fn_prime - derivative of relevant activation function

        assert!(
            targets.rows == 1,
            "back propagate, targets must be a matrix of form 1 x samples "
        );

        // # output layer
        matrix::get_softmax_delta(
            &mut self.deltas[self.layers - 1],
            &self.batch_activations[self.layers],
            &targets,
        );
        matrix::multABT(
            &self.deltas[self.layers - 1],
            &self.batch_activations[self.layers - 1],
            &mut self.derivative_W[self.layers - 1],
        );
        matrix::bias_from_delta(
            &mut self.derivative_b[self.layers - 1],
            &self.deltas[self.layers - 1],
        );

        // # hidden layers
        for temp_idx in 0..self.layers - 1 {
            let idx = self.layers - 2 - temp_idx;
            matrix::multATB(
                &self.W[idx + 1],
                &self.deltas[idx + 1],
                &mut self.backprop_m_holder[idx],
            );
            matrix::e_apply_function(
                &mut self.backprop_z_holder[idx],
                &self.batch_potentials[idx + 1],
                hidden_activation_fn_prime,
            );
            matrix::e_multAB(
                &mut self.deltas[idx],
                &self.backprop_m_holder[idx],
                &self.backprop_z_holder[idx],
            );

            matrix::multABT(
                &self.deltas[idx],
                &self.batch_activations[idx],
                &mut self.derivative_W[idx],
            );
            matrix::bias_from_delta(&mut self.derivative_b[idx], &self.deltas[idx]);
        }
    }

    fn initialize_test_matrices(&mut self, test_size: usize) {
        /// Method to create matrices for storing test predictions
        for layer in self.batch_activations.iter() {
            self.test_activations
                .push(Matrix::new(layer.rows, test_size));
            self.test_potentials
                .push(Matrix::new(layer.rows, test_size));
        }
    }

    fn adam_update(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        friction: f32,
        L2_rate: f32,
        batch_size: usize,
    ) {
        /// Method to calculate and update parameters relevant to Adam optimizer
        for idx in 0..self.layers {
            for row in 0..self.W[idx].rows {
                for col in 0..self.W[idx].columns {
                    self.velocities_W[idx].data[row][col] = momentum
                        * self.velocities_W[idx].data[row][col]
                        + (1.0 - momentum) * self.derivative_W[idx].data[row][col];
                    self.summation_W[idx].data[row][col] = friction
                        * self.summation_W[idx].data[row][col]
                        + (1.0 - friction) * self.derivative_W[idx].data[row][col].powf(2.0);
                    self.W[idx].data[row][col] = (1.0
                        - learning_rate * L2_rate / batch_size as f32)
                        * self.W[idx].data[row][col]
                        - (learning_rate / (self.summation_W[idx].data[row][col] + 0.00001).sqrt())
                            * self.velocities_W[idx].data[row][col];
                }
                self.velocities_b[idx].data[row][0] = momentum
                    * self.velocities_b[idx].data[row][0]
                    + (1.0 - momentum) * self.derivative_b[idx].data[row][0];
                self.summation_b[idx].data[row][0] = friction * self.summation_b[idx].data[row][0]
                    + (1.0 - friction) * self.derivative_b[idx].data[row][0].powf(2.0);
                self.b[idx].data[row][0] = (1.0 - learning_rate * L2_rate / batch_size as f32)
                    * self.b[idx].data[row][0]
                    - (learning_rate / (self.summation_b[idx].data[row][0] + 0.00001).sqrt())
                        * self.velocities_b[idx].data[row][0];
            }
        }
    }

}

fn get_batch_columns(batch: &mut Matrix, dataX: &Matrix, inds: &Matrix) {
    /// Method that copies given columns (with indices inds[index]) into batch
    /// # Arguments
    /// param:batch - output matrix, batch.rows=inds.columns, batch.columns=dataX.columns
    /// param:dataX - input data from which it creates
    /// param:ids - indices of rows that should be sampled to batch

    assert!(batch.columns == inds.columns);
    assert!(batch.rows == dataX.rows);

    for r in 0..batch.rows {
        for i in 0..inds.columns {
            let c = inds.data[0][i];
            batch.data[r][i] = dataX.data[r][c as usize];
        }
    }
}

fn main() {
    let mut fashion_mnist_train_vectors_path = "";
    let mut fashion_mnist_train_labels_path = "";
    let mut fashion_mnist_test_vectors_path = "";
    let mut test_predictions_path = "";

    // assigninng the correct paths to files
    if cfg!(target_os = "linux") {
        fashion_mnist_train_vectors_path = "data/fashion_mnist_train_vectors.csv";
        fashion_mnist_train_labels_path = "data/fashion_mnist_train_labels.csv";
        fashion_mnist_test_vectors_path = "data/fashion_mnist_test_vectors.csv";
        test_predictions_path = "actualPredictions"

    } else {
        fashion_mnist_train_vectors_path = r"C:\Users\capek\Desktop\School\Neural networks\pv021\project\data\fashion_mnist_train_vectors.csv";
        fashion_mnist_train_labels_path = r"C:\Users\capek\Desktop\School\Neural networks\pv021\project\data\fashion_mnist_train_labels.csv";
        fashion_mnist_test_vectors_path = r"C:\Users\capek\Desktop\School\Neural networks\pv021\project\data\fashion_mnist_test_vectors.csv";
    }

    // # variables
    let batch_size = 64;
    let total = 60000;
    let val_size = (0.1 * (total as f32)).floor() as usize;
    let train_size = total - val_size;

    let mut split = generate_train_validation_split(total, val_size);
    let (trainX, validationX) =
        read_csv_with_split(fashion_mnist_train_vectors_path, 28 * 28, 60000, &split);
    let (trainY, validationY) =
        read_csv_with_split(fashion_mnist_train_labels_path, 1, 60000, &split);

    let testX = read_csv(fashion_mnist_test_vectors_path, 10000, 28 * 28);
    
    let mut predictions = Matrix::new(1, batch_size);
    let mut network = Network::new(28 * 28, vec![128, 64, 10], batch_size, val_size);
    let iterations = usize::from(train_size / batch_size.clone()) * 10;
    let mut learning_rate = 0.0005;
    let momentum = 0.6;
    let friction = 0.99;
    let L2_rate = 0.96;
    let keep_prob = 1.0;
    let mut batch_data = Matrix::new(28 * 28, batch_size);
    let mut batch_targets = Matrix::new(1, batch_size);
    
    let mut val_predictions = Matrix::new(validationY.rows, validationY.columns);
    let mut test_predictions = Matrix::new(1, testX.columns);

    // # training phase
    for i in 0..iterations {
        let batch_indices = generate_indices_for_batch(batch_size, train_size);
        get_batch_columns(&mut batch_data, &trainX, &batch_indices);
        get_batch_columns(&mut batch_targets, &trainY, &batch_indices);

        network.feed_forward(&batch_data, &relu, keep_prob);
        predictions.predict_from_softmax(&network.batch_activations[network.layers]);
        network.back_propagate(&batch_targets, &relu_prime);
        network.adam_update(learning_rate, momentum, friction, L2_rate, batch_size);

        if i % 1000 == 0 {
            /*time*/
            println!("New Epoch:\t{}\n", i as f32 / 1000.0);
        }

        if i % 1000 == 999 {
            /*time*/
            println!("Calculating validation set ...");
            network.feed_forward(&validationX, &relu, 0.0);
            val_predictions.predict_from_softmax(&network.val_activations[network.layers]);
            let val_acc = get_acc(&val_predictions, &validationY);
            println!("validation accuracy:\t{}\n", val_acc);
            if val_acc >= 90.0 {
                break;
            }
        }
    }

    // # prediction phase
    println!("\n\n Calculating test set ...");
    network.feed_forward(&testX, &relu, 0.0);
    test_predictions.predict_from_softmax(&network.test_activations[network.layers]);
    write_csv(test_predictions_path, &test_predictions);
}

// ================ TESTS ==================

#[test]
fn test_feed_forward_xor() {
    let mut network = Network::new(2, vec![2, 2], 4, 0);
    network.W[0].data[0] = vec![2.0, 2.0];
    network.W[0].data[1] = vec![-2.0, -2.0];

    network.W[1].data[0] = vec![-1.0, -1.0];
    network.W[1].data[1] = vec![1.0, 1.0];

    network.b[0].data[0][0] = -1.0;
    network.b[0].data[1][0] = 3.0;

    network.b[1].data[0][0] = 1.0;
    network.b[1].data[1][0] = -2.0;

    
    let mut testX = Matrix::new(2, 4);
    testX.data[0] = vec![1.0, 1.0, 0.0, 0.0];
    testX.data[1] = vec![1.0, 0.0, 1.0, 0.0];
    let mut predictions = Matrix::new(1, 4);

    network.feed_forward(&testX, &relu_prime, 0.0);
    predictions.predict_from_softmax(&network.batch_activations[network.layers]);
    
    assert_eq!(predictions.data[0][0], 0.0); // 1 ^ 1 = 0
    assert_eq!(predictions.data[0][1], 1.0); // 1 ^ 0 = 1
    assert_eq!(predictions.data[0][2], 1.0); // 0 ^ 1 = 1
    assert_eq!(predictions.data[0][3], 0.0); // 0 ^ 0 = 0
}


#[test]
fn test_sigmoid() {
    let input1 = 1.0;
    let output1 = 0.7310585786300048792512;
    let input2 = 0.5;
    let output2 = 0.6224593312018545646389;
    let input3 = 0.0;
    let output3 = 0.5;
    let input4 = -0.23;
    let output4 = 0.4427521454014444228833;
    let input5 = -1.23;
    let output5 = 0.2261814257305461957548;

    assert_eq!(sigmoid(input1), output1);
    assert_eq!(sigmoid(input2), output2);
    assert_eq!(sigmoid(input3), output3);
    assert_eq!(sigmoid(input4), output4);
    assert_eq!(sigmoid(input5), output5);
}
