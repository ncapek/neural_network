extern crate rand;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::fs::File;
use std::time::{Duration, Instant};
use rand::prelude::*;
use relative_path::RelativePath;
use std::path::Path;
use std::fs;
use std::path::PathBuf;

mod matrix;
use crate::matrix::Matrix;

fn sigmoid (input: f32) -> f32 {
    1.0 / (1.0 + (-input).exp())
}

fn sigmoid_prime (input: f32) -> f32 {
    sigmoid(input)*(1.0-sigmoid(input))
}

fn relu (input: f32) -> f32 {
    0.0_f32.max(input)    
}

fn relu_prime (input: f32) -> f32 {
    if input >= 0.0 {
        1.0
    } else {
        0.0
    }
}

fn get_acc (predictions: &Matrix, targets: &Matrix) -> f32 {
    // from row vectors
    let num_of_samples = targets.columns as f32;
    let mut error = 0.0;
    for sample in 0..targets.columns {
        if (targets.data[0][sample] - predictions.data[0][sample]) == 0.0 {
            error = error + 1.0;
        }
    }
    error/num_of_samples
}

fn generate_train_validation_split() -> Vec<usize> {
    let mut vec = Vec::with_capacity(15000);
    for i in 45000..60000{
        vec.push(i);
    }

    let mut rng = rand::thread_rng();
    let mut ind = 0;
    for number in 0..60000 {
        if rng.gen::<f32>() < 0.25 {
            vec[ind] = number;
            ind += 1;
        }
        if 60000 - number <= 15000 - ind || ind >= 15000{
            println!("number: {:?}, ind: {:?}", number, ind);
            break;
        } 
    }
    vec   
} 


fn read_csv(path: &str, n_rows: usize, n_cols: usize) -> Matrix {
    let mut output_tr = Matrix::new(n_cols, n_rows);
    
	let f = BufReader::new(File::open(path).expect("open failed"));
    //let mut row = 0;
    let mut col = 0;
    for line in f.lines() {
        //row = 0;
		let mut row = 0;
        for c in line.expect("lines failed").chars() {
            match c {
                ',' => {output_tr.data[row][col] = output_tr.data[row][col] / 255.0; row += 1},
                _ => output_tr.data[row][col] = output_tr.data[row][col] * 10.0 + (c as i32 - 48) as f32,
            }
        }
        col += 1;
    }
    output_tr
}


fn read_csv_with_split(path: &str, n_rows: usize, split: &Vec<usize>) -> (Matrix, Matrix) {
    let mut train = Matrix::new(n_rows, 45000);
    let mut validate = Matrix::new(n_rows, 15000);
    
    let f = BufReader::new(File::open(path).expect("open failed"));

    let mut ind = 0;
    let mut col = 0;

    for line in f.lines() {
        if ind < 15000 && col == split[ind] {
            let mut row = 0;
            for c in line.expect("lines failed").chars() {
                match c {
                    ',' => {validate.data[row][ind] = validate.data[row][ind] / 255.0; row += 1},
                    _ => validate.data[row][ind] = validate.data[row][ind] * 10.0 + (c as i32 - 48) as f32,
                }
            }
            if ind < 15000 {
                ind += 1;
            }            
        }   else {
            let mut row = 0;
            for c in line.expect("lines failed").chars() {
                match c {
                    ',' => {train.data[row][col-ind] = train.data[row][col-ind] / 255.0; row += 1},
                    _ => train.data[row][col-ind] = train.data[row][col-ind] * 10.0 + (c as i32 - 48) as f32,
                }
            }
        }
        col += 1;

    }
    (train, validate)
}

fn write_csv(path: &str, m: &Matrix) {
    let f = File::create(path).unwrap();
    let mut wrt = BufWriter::new(f);
    let mut num = 2;

    for colrow in 0..m.columns {
        while num >= 0 {
            //let mut ch = (m.data[0][colrow] / (10.0_f32.powi(num))) as i32;
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


fn generate_indeces_for_batch(batch_size: usize, batch_number: usize, max_number: usize) -> Matrix {
    // The function generates indeces of rows/columns that should be part of a batch
    // In each line of output are indeces for one batch.
    // param:batch_size - size of one batch
    // param:batch_number - number of batches
    // param:max_number - maximal value of generated indices (number of columns/rows)
    //TODO jak moc vadi, ze se indexy muzou opakovat?
    let mut rng = rand::thread_rng();
    let mut indices = Matrix::new(batch_number, batch_size);
    for r in 0..batch_number {
        for c in 0..batch_size {
            indices.data[r][c] = rng.gen_range(0, max_number) as f32;
        }
    }
    indices
}

#[derive(Debug)]
struct Network {
    // in this class, we are always assuming input_data matrix is of the form features x samples
    // input_dim - takes the dimensionality of input features, in this project always 28*28
    // batch_size - takes the number of samples in batch dataset
	// val_size - takes the number of samples in validation dataset
	// test_size - takes the number of samples in test dataset
    // layers - takes the number of layers
    // sizes - takes a vector containing the number of neurons in each layer, dimensions: 1xlayers
    // W - instantiates vector containing matrix of weights between the previous and current layer, dimensions: 1xlayers
    // b - instantiates vector of single column matrices containing bias terms for each layer, dimensions: 1xlayers
    // activations - instantiates vector containing matrix of activations for given layer, dimensions: 1x(layers+1)
    // potentials - instantiates vector containing matrix of activations for given layer, dimensions: 1x(layers+1)
    // derivative_W - instantiates vector containing matrix of weight derivatives, dimensions: 1x(layers)
    // derivative_b - instantiates vector of single column matrices containing bias derivatives, dimensions: 1x(layers)
    // deltas - instantiates vector containing matrix of deltas, dimensions: 1x(layers)
    // velocities - instantiates vector containing matrix of velocities, dimensions: 1xlayers
    // summation - instantiates vector containing matrix of summations from RMSprob, dimensions: 1xlayers
	// summation - instantiates vector containing matrix holders for temporary calculations in backpropagation step, dimensions: 1xlayers

    input_dim: usize,
    batch_size: usize, 
	val_size: usize, 
	//test_size: usize, 
    layers: usize,
    sizes: Vec<usize>,
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
    backprop_z_holder: Vec<Matrix>
}

impl Network {
    fn new(input_dim: usize, sizes: Vec<usize>, batch_size: usize, val_size: usize) -> Network {

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
            // normalized intitiliazation
            // let lower = -(2.0/(previous_layer_size as f32 + layer_size.clone() as f32)).sqrt();        
             let upper = (2.0/(previous_layer_size.clone() as f32)).sqrt(); 
            // glorot-bengio
            // let lower = -(1.0/(previous_layer_size as f32)).sqrt();        
             //let var = (1.0/(previous_layer_size as f32)).sqrt(); 
            // currently He


            //W.push(Matrix::randu_new(layer_size.clone(), previous_layer_size.clone(), lower, upper));
            W.push(Matrix::randn_new(layer_size.clone(), previous_layer_size.clone(), 0.0, upper));
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
			
			if layer_idx < sizes.len()-1 {
				backprop_m_holder.push(Matrix::new(deltas[layer_idx].rows, deltas[layer_idx].columns));
			    backprop_z_holder.push(Matrix::new(batch_potentials[layer_idx+1].rows, batch_potentials[layer_idx+1].columns));
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
			backprop_z_holder: backprop_z_holder
        }
    }

    fn feed_forward (&mut self, data: &Matrix, activation_fn: &dyn Fn(f32) -> f32, keep_prob: f32) {
        // data: Matrix of input data features x samples
        // activation_fn: activation function which will be applied to all hidden layers
        // output_activation_fn: activation function which will be applied to output layer 
        // output values: Vector containing matrices with inner potentials, Vector containing matrices of activations, Matrix containing outputs
		
		// training branch
		if data.columns == self.batch_size {
			self.batch_activations[0] = data.clone();
			for layer in 0..self.layers {
				self.batch_potentials[layer+1].multAB(&self.W[layer], &self.batch_activations[layer]);
				self.batch_potentials[layer+1].c_add_vector(&self.b[layer]);
				
				if layer == self.layers - 1 {
					self.batch_activations[layer+1].c_apply_softmax(&self.batch_potentials[layer+1]);
				} else {
					matrix::e_apply_function(&mut self.batch_activations[layer+1], &self.batch_potentials[layer+1], &activation_fn);
					//self.apply_dropout(layer, keep_prob);
				}
			}	 
		}
		// val prediction branch
		else if data.columns == self.val_size {
			self.val_activations[0] = data.clone();
			for layer in 0..self.layers {
				self.val_potentials[layer+1].multAB(&self.W[layer], &self.val_activations[layer]);
				self.val_potentials[layer+1].c_add_vector(&self.b[layer]);	
				
				if layer == self.layers - 1 {
					self.val_activations[layer+1].c_apply_softmax(&self.val_potentials[layer+1]);
				} else {
					matrix::e_apply_function(&mut self.val_activations[layer+1], &self.val_potentials[layer+1], &activation_fn);
				}
			}
		} 
		
		// test prediction branch
		else {
            self.initialize_test_matrices(data.columns);
			self.test_activations[0] = data.clone();
            for layer in 0..self.layers {
                self.test_potentials[layer+1].multAB(&self.W[layer], &self.test_activations[layer]);
                self.test_potentials[layer+1].c_add_vector(&self.b[layer]);  
                
                if layer == self.layers - 1 {
                    self.test_activations[layer+1].c_apply_softmax(&self.test_potentials[layer+1]);
                } else {
                    matrix::e_apply_function(&mut self.test_activations[layer+1], &self.test_potentials[layer+1], &activation_fn);
                }
            }
		}
	}

    fn back_propagate (&mut self, targets: &Matrix, hidden_activation_fn_prime: &dyn Fn(f32) -> f32) {
        // targets must of matrix of form 1 x samples

        assert!(targets.rows == 1, "back propagate, targets must of matrix of form 1 x samples ");
        matrix::get_softmax_delta(&mut self.deltas[self.layers-1], &self.batch_activations[self.layers], &targets);
        matrix::multABT(&self.deltas[self.layers-1], &self.batch_activations[self.layers-1], &mut self.derivative_W[self.layers-1]);
        matrix::bias_from_delta(&mut self.derivative_b[self.layers-1], &self.deltas[self.layers-1]);
        
        // hidden layers
        for temp_idx in 0..self.layers - 1 {
            let idx = self.layers - 2 - temp_idx;
            matrix::multATB(&self.W[idx+1], &self.deltas[idx+1], &mut self.backprop_m_holder[idx]);
            matrix::e_apply_function(&mut self.backprop_z_holder[idx], &self.batch_potentials[idx+1], hidden_activation_fn_prime); 
            matrix::e_multAB(&mut self.deltas[idx], &self.backprop_m_holder[idx], &self.backprop_z_holder[idx]);

            matrix::multABT(&self.deltas[idx], &self.batch_activations[idx], &mut self.derivative_W[idx]);
            matrix::bias_from_delta(&mut self.derivative_b[idx], &self.deltas[idx]);
        }              
    }
	
	fn initialize_test_matrices(&mut self, test_size: usize) {
		for layer in self.batch_activations.iter() {
            self.test_activations.push(Matrix::new(layer.rows, test_size));
            self.test_potentials.push(Matrix::new(layer.rows, test_size));
        }
	}

    fn adam_update (&mut self, learning_rate: f32, momentum: f32, friction: f32, L2_rate: f32, batch_size: usize) {
        for idx in 0..self.layers { 
			for row in 0..self.W[idx].rows {
				for col in 0..self.W[idx].columns {
					self.velocities_W[idx].data[row][col] = momentum * self.velocities_W[idx].data[row][col] + (1.0 - momentum) * self.derivative_W[idx].data[row][col];					
					self.summation_W[idx].data[row][col] = friction * self.summation_W[idx].data[row][col] + (1.0 - friction) * self.derivative_W[idx].data[row][col].powf(2.0);					
					self.W[idx].data[row][col] = (1.0 - learning_rate * L2_rate / batch_size as f32) * self.W[idx].data[row][col] - (learning_rate / (self.summation_W[idx].data[row][col] + 0.00001).sqrt()) * self.velocities_W[idx].data[row][col];					
				}
				self.velocities_b[idx].data[row][0] = momentum * self.velocities_b[idx].data[row][0] + (1.0 - momentum) * self.derivative_b[idx].data[row][0];
				self.summation_b[idx].data[row][0] = friction * self.summation_b[idx].data[row][0] + (1.0 - friction) * self.derivative_b[idx].data[row][0].powf(2.0);
				self.b[idx].data[row][0] = (1.0 - learning_rate * L2_rate / batch_size as f32) * self.b[idx].data[row][0] - (learning_rate / (self.summation_b[idx].data[row][0] + 0.00001).sqrt()) * self.velocities_b[idx].data[row][0];
			}
        }			
    }
	
	fn apply_dropout(&mut self, layer_idx: usize, keep_prob: f32) {
		let mut rng = rand::thread_rng();
		for row in 0..self.batch_activations[layer_idx+1].rows {
			for column in 0..self.batch_activations[layer_idx+1].columns {
				if rng.gen_range(0.0, 1.0) > keep_prob {
					self.batch_activations[layer_idx+1].data[row][column] = 0.0;
				} else {
					self.batch_activations[layer_idx+1].data[row][column] /= keep_prob;
				}
			}
		}	
   	}
}

fn get_batch_rows(batch : &mut Matrix, dataX : &Matrix, inds : &Matrix, index: usize) {
    // batch ... output matrix, batch.rows=inds.columns, batch.columns=dataX.columns
    // dataX ... input data from which it creates
    // ids ... indices of rows that should be sampled to batch
    assert!(batch.rows == inds.columns);
    assert!(batch.columns == dataX.columns);
    
    for i in 0..inds.columns {
        let row = inds.data[index][i];
        batch.data[i] = dataX.data[row as usize].clone();
    }
}

fn get_batch_columns(batch : &mut Matrix, dataX : &Matrix, inds : &Matrix, index: usize) {
    // batch ... output matrix, batch.rows=inds.columns, batch.columns=dataX.columns
    // dataX ... input data from which it creates
    // ids ... indices of rows that should be sampled to batch
    assert!(batch.columns == inds.columns);
    assert!(batch.rows == dataX.rows);
    
    for r in 0..batch.rows {
        for i in 0..inds.columns {
            let c = inds.data[index][i];
            batch.data[r][i] = dataX.data[r][c as usize];
        }    
    }
    
}

fn main() {
	/*time*/
	let now = Instant::now();
	
    let mut fashion_mnist_train_vectors_path = "";
    let mut fashion_mnist_train_labels_short_path = "";
    let mut fashion_mnist_train_vectors_short_path = "";
    let mut fashion_mnist_train_labels_path = "";
    let mut fashion_mnist_test_vectors_path = "";
    let mut fashion_mnist_test_labels_path = "";
        
    let mut train_predictions_path = "";
    let mut test_predictions_path = "";

    let mut xor_vectors_path = "";
    let mut xor_labels_path = "";


    // assing the correct paths to files
    if cfg!(target_os = "linux") {
        println!("LINUX part");
        fashion_mnist_train_vectors_path = "./src/fashion_mnist_train_vectors.csv";
        fashion_mnist_train_labels_path = "./src/fashion_mnist_train_labels.csv";
        fashion_mnist_test_vectors_path = "./src/fashion_mnist_test_vectors.csv";
        fashion_mnist_test_labels_path = "./src/fashion_mnist_test_labels.csv";
        xor_vectors_path = "./src/xor_vectors.csv";
        xor_labels_path = "./src/xor_labels.csv";

    } else {		
        fashion_mnist_train_vectors_path = r"src\fashion_mnist_train_vectors.csv";
		fashion_mnist_train_labels_path = r"src\fashion_mnist_train_labels.csv";
		fashion_mnist_test_vectors_path = r"src\fashion_mnist_test_vectors.csv";
		fashion_mnist_test_labels_path = r"src\fashion_mnist_test_labels.csv";
		xor_vectors_path = r"src\xor_vectors.csv";
		xor_labels_path = r"src\xor_labels.csv";

    }


	let batch_size = 64;
	let val_size = 15000;
	let train_size = 60000-val_size;
	
    let mut split = generate_train_validation_split();
    let (mut trainX, mut validationX) = read_csv_with_split(fashion_mnist_train_vectors_path, 28*28, &split);
    let (mut trainY, mut validationY) = read_csv_with_split(fashion_mnist_train_labels_path, 1, &split);

	let mut testX = read_csv(fashion_mnist_test_vectors_path, 10000, 28*28);
	let testY = read_csv(fashion_mnist_test_labels_path, 10000, 1);

	let mut predictions = Matrix::new(1, batch_size);
	let mut network = Network::new(28*28, vec![128,128,10], batch_size, val_size);
	let iterations = 10000;
    let mut learning_rate = 0.003;
	let decay = learning_rate/iterations as f32;
	let momentum = 0.99;
	let friction = 0.99;
	let L2_rate = 0.9;
	let keep_prob = 0.93;
	let mut batch_data = Matrix::new(28*28, batch_size);
	let mut batch_targets = Matrix::new(1, batch_size);
	let mut batch_targets = Matrix::new(1, batch_size);
	let batch_indices = generate_indeces_for_batch(batch_size, iterations, train_size);
	let mut val_predictions = Matrix::new(validationY.rows, validationY.columns);
	let mut test_predictions = Matrix::new(testY.rows, testY.columns);
	get_batch_columns(&mut batch_data, &trainX, &batch_indices, 0);
	get_batch_columns(&mut batch_targets, &trainY, &batch_indices, 0);
	
	for i in 0..iterations {
		get_batch_columns(&mut batch_data, &trainX, &batch_indices, i);
		get_batch_columns(&mut batch_targets, &trainY, &batch_indices, i);

		network.feed_forward(&batch_data, &relu, keep_prob);
		predictions.predict_from_softmax(&network.batch_activations[network.layers]);    
		network.back_propagate(&batch_targets, &relu_prime);
		network.adam_update(learning_rate, momentum, friction, L2_rate, batch_size);

        if i % 1000 == 0{
			/*time*/ println!("Time elapsed:\t{}s\n", now.elapsed().as_secs());
			println!("New Epoch:\t{}\n", i as f32 / 1000.0);
		}

		if i % 1000 == 999 {
			/*time*/ println!("Time elapsed:\t{}s\n", now.elapsed().as_secs());
			println!("Calculating validation set ...\n");
			network.feed_forward(&validationX, &relu, 0.0);
			val_predictions.predict_from_softmax(&network.val_activations[network.layers]);
			let val_acc = get_acc(&val_predictions, &validationY);
			println!("validation accuracy:\t{}\n", val_acc);
			if val_acc >= 90.0 {
				break;
			}
		}

/*
		if i % 25 == 0 {
			//println!("gradients:\t{:?}\n", network.derivative_W);
			//println!("predictions:\t{:?}\n", predictions);
			println!("batch accuracy:\t{}\n", get_acc(&predictions, &batch_targets));
		}
*/

		//learning_rate = learning_rate * (1.0 / (1.0 + decay * i as f32))
	} 
	println!("\n\n Calculating test set ...\n");
	network.feed_forward(&testX, &relu, 0.0);
	test_predictions.predict_from_softmax(&network.test_activations[network.layers]);
	println!("Test set accuracy:\t{}\n", get_acc(&test_predictions, &testY));
}







// ================ TESTS ==================

#[test]
fn test_feed_forward_xor () {
    let mut network = Network::new(2, vec![2, 2], 4, 0);
    network.W[0].data[0] = vec![2.0, 2.0];
    network.W[0].data[1] = vec![-2.0, -2.0];

    network.W[1].data[0] = vec![-1.0, -1.0];
    network.W[1].data[1] = vec![1.0, 1.0];
    
    network.b[0].data[0][0] = -1.0;
    network.b[0].data[1][0] = 3.0;

    network.b[1].data[0][0] = 1.0;
    network.b[1].data[1][0] = -2.0;

    println!("{:?}", network);
    let mut testX = Matrix::new(2,4);
    testX.data[0] = vec![1.0, 1.0, 0.0, 0.0];
    testX.data[1] = vec![1.0, 0.0, 1.0, 0.0];
    let mut predictions = Matrix::new(1,4);

    network.feed_forward(&testX, &relu_prime, 0.0);
    predictions.predict_from_softmax(&network.batch_activations[network.layers]);
    println!("{:?}", predictions);
    assert_eq!(predictions.data[0][0], 0.0);   // 1 ^ 1 = 0
    assert_eq!(predictions.data[0][1], 1.0);   // 1 ^ 0 = 1
    assert_eq!(predictions.data[0][2], 1.0);   // 0 ^ 1 = 1
    assert_eq!(predictions.data[0][3], 0.0);   // 0 ^ 0 = 0
}

/*
#[test]
fn test_linear_prediction () {
    let mut trainX = Matrix::new(2,10);
    trainX.data[0] = vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 1.0, 3.0, 2.0, 3.0,];
    trainX.data[1] = vec![4.0, 5.0, 6.0, 1.0, 2.0, 1.0, 3.0, 6.0, 1.0, 1.0];

    let mut network = Network::new(2, 1, vec![2], 10, 0);
    let mut learning_rate = 0.01;
    let mut predictions = Matrix::new(1,10);
    let mut trainY = Matrix::new(1,10);
    trainY.data[0] = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

    for i in 1..20 {
        network.feed_forward(&trainX, &relu, 0.0);
        predictions.predict_from_softmax(&network.batch_activations[network.layers]);
        network.back_propagate(&trainY, &relu_prime);
        
        for idx in 0..network.layers { 
            network.W[idx].classic_gradient_descent(&network.derivative_W[idx], learning_rate);
            network.b[idx].classic_gradient_descent(&network.derivative_b[idx], learning_rate);
        }  
    }    
    assert_eq!(predictions.data[0], trainY.data[0]);          
}


#[test]
fn test_xor_learning () {
    let mut trainX = Matrix::new(2,4);
    trainX.data[0] = vec![1.0, 1.0, 0.0, 0.0];
    trainX.data[1] = vec![0.0, 1.0, 1.0, 0.0];

    let mut network = Network::new(2, 2, vec![2,2], 4, 0);
    let mut learning_rate = 0.2;
    let mut predictions = Matrix::new(1,4);
    let mut trainY = Matrix::new(1,4);
    trainY.data[0] = vec![1.0, 0.0, 1.0, 0.0];
    let mut accuracy = 0.0;
    let mut iter = 0;

    while accuracy < 1.0 && iter < 30{
        iter = iter + 1;
        network = Network::new(2, 2, vec![2,2], 4, 0);
        println!("{:?}", iter);
        for i in 1..80 {
        
        network.feed_forward(&trainX, &relu, 0.0);
        predictions.predict_from_softmax(&network.batch_activations[network.layers]);
        network.back_propagate(&trainY, &relu_prime);
        accuracy = get_acc(&predictions, &trainY);
        if accuracy > 0.9 {
            break;
        } 
        for idx in 0..network.layers { 
            network.W[idx].classic_gradient_descent(&network.derivative_W[idx], learning_rate);
            network.b[idx].classic_gradient_descent(&network.derivative_b[idx], learning_rate);
        }  
    }    
        println!("{:?}", accuracy);
}
    assert_eq!(predictions.data[0], trainY.data[0]);          
}
*/

#[test]
fn test_get_batch_rows() {
    let mut dataX = Matrix::new(10,4);
    for i in 0..dataX.rows {
        for j in 0..dataX.columns {
            dataX.data[i][j] = (i + j) as f32;
        }
    }

    let mut inds = Matrix::new(2,5);
    inds.data[0] = vec![1.0, 4.0, 7.0, 8.0, 5.0];
    inds.data[1] = vec![9.0, 3.0, 2.0, 0.0, 6.0];
    let mut batch = Matrix::new(5,4);

    get_batch_rows(&mut batch, &dataX, &inds, 0);

    for i in 0..5 {
        assert_eq!(dataX.data[inds.data[0][i] as usize], batch.data[i]);
    }

    get_batch_rows(&mut batch, &dataX, &inds, 1);
    for i in 0..5 {
        assert_eq!(dataX.data[inds.data[1][i] as usize], batch.data[i]);
    }
}


#[test]
fn test_get_batch_columns() {
    let mut dataX = Matrix::new(4,10);
    for i in 0..dataX.rows {
        for j in 0..dataX.columns {
            dataX.data[i][j] = (i + j) as f32;
        }
    }

    let mut inds = Matrix::new(2,5);
    inds.data[0] = vec![1.0, 4.0, 7.0, 8.0, 5.0];
    inds.data[1] = vec![9.0, 3.0, 2.0, 0.0, 6.0];
    let mut batch = Matrix::new(4,5);

    get_batch_columns(&mut batch, &dataX, &inds, 0);
    for c in 0..5 {
        for r in 0..4 {
            assert_eq!(dataX.data[r][inds.data[0][c] as usize], batch.data[r][c]);
        }
    }

    get_batch_columns(&mut batch, &dataX, &inds, 1);
    for c in 0..5 {
        for r in 0..4 {
            assert_eq!(dataX.data[r][inds.data[1][c] as usize], batch.data[r][c]);
        }
    }

}

#[test]
fn test_sigmoid () {
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



