use rand::prelude::*;
use rand::distributions::{Normal, Distribution};
use rayon::prelude::*;

#[derive(Debug)]
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<Vec<f32>>

}
	impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        Matrix {
            rows: rows,
            columns: columns,
            data: vec![vec![0.0; columns];rows]
        }
    }

    pub fn ones_new(rows: usize, columns: usize) -> Matrix {
        Matrix {
            rows: rows,
            columns: columns,
            data: vec![vec![1.0; columns];rows]
        }
    }

    pub fn randn_new(rows: usize, columns: usize, mean: f32, var: f32) -> Matrix {
        // initiliazes value from N(mean, var)
        let normal = Normal::new(mean as f64, 1.0);
        let mut data = Vec::new();
        for _r in 0..rows {
            let mut row_data = Vec::new();
            for _c in 0..columns {
                //row_data.push(rng.gen_range(lower, upper));
                row_data.push((normal.sample(&mut rand::thread_rng()) as f32)*var);
            }
            data.push(row_data);
        }

        Matrix {
            rows: rows,
            columns: columns,
            data: data
        }
    }


    pub fn randu_new(rows: usize, columns: usize, lower: f32, upper: f32) -> Matrix {
        // initiliazes value from U(lower, upper)
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        for _r in 0..rows {
            let mut row_data = Vec::new();
            for _c in 0..columns {
                row_data.push(rng.gen_range(lower, upper));
            }
            data.push(row_data);
        }

        Matrix {
            rows: rows,
            columns: columns,
            data: data
        }
    }
	
	pub fn copy_data(&mut self, m: &Matrix) {
		assert!(self.rows==m.rows, "self.rows==m.rows in matrix::copy_data");
        assert!(self.columns==m.columns, "self.columns==m.columns in matrix::copy_data");
		for row in 0..m.rows {
			for column in 0..m.columns {
				self.data[row][column] = m.data[row][column];
			}
		}
	}


    pub fn transpose(&self, output: &mut Matrix) {
        assert!(self.rows==output.columns, "self.rows==output.columns in matrix::transpose");
        assert!(self.columns==output.rows, "self.columns==output.rows in matrix::transpose");

        for i in 0..output.rows {
            for j in 0..output.columns {
                output.data[i][j] = self.data[j][i];
            }
        }
    }
/*	
	pub fn multAB(&mut self, m1: &Matrix, m2: &Matrix) {
		assert!(m1.columns==m2.rows, "m1.columns==m2.rows in matrix::multAB");
		assert!(m1.rows==self.rows, "m1.rows==self.rows in matrix::multAB");
		assert!(m2.columns==self.columns, "m2.columns==self.columns in matrix::multAB");

        for i in 0..m1.rows {
            for j in 0..m2.columns {
				self.data[i][j] = 0.0;
            }
        }
		for i in 0..m1.rows {
			for k in 0..m1.columns {
			     for j in 0..m2.columns {
					self.data[i][j] += m1.data[i][k]*m2.data[k][j];
				}
			}
		}
	}
*/

    pub fn multAB(&mut self, m1: &Matrix, m2: &Matrix) {
        assert!(m1.columns==m2.rows, "m1.columns==m2.rows in matrix::multAB");
        assert!(m1.rows==self.rows, "m1.rows==self.rows in matrix::multAB");
        assert!(m2.columns==self.columns, "m2.columns==self.columns in matrix::multAB");

        let mut unordered_rows = (0..m1.rows)
            .into_par_iter()
            .map(move |i| {
            let m1_row =  &m1.data[i];

            (i, (0..m2.columns)
                .map(|j| (0..m2.rows).map(|k| m1_row[k] * m2.data[k][j]).sum())
                .collect::<Vec<f32>>())
        })
        .collect::<Vec<(usize, Vec<f32>)>>();

        unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));
        self.data = unordered_rows.into_iter().map(|(_, row)| row).collect();
    }


    //pub fn mult_f32(&mut self, c: f32) {
    pub fn e_mult_f32(&mut self, c: f32) {
        // TODO not used ??
        for row in 0..self.rows {
            for column in 0..self.columns {
                self.data[row][column] *= c;
            }
        }
    }


/*
    pub fn inplace_multcA(output: &mut Matrix, c: f32) {
        for i in 0..output.rows {
            for j in 0..output.columns {
                output.data[i][j] = output.data[i][j]*c;
            }
        }
    }
*/
    //inplace_e_multAB
    pub fn e_mult_matrix(&mut self, m: &Matrix) {
        // TODO - not used ?
        assert!(self.rows==m.rows, "self.rows==m.rows in matrix::e_mult_matrix");
        assert!(self.columns==m.columns, "self.columns==m.columns in matrix::e_mult_matrix");
        for row in 0..self.rows {
            for column in 0..self.columns {
                self.data[row][column] = self.data[row][column] * m.data[row][column];
            }
        }
    }


    //    pub fn inplace_c_add_vector(output: &mut Matrix, m: &Matrix) {
    pub fn c_add_vector2(&mut self, m: &Matrix) {
        assert!(m.rows == self.rows, "m.rows == output.rows in matrix::c_add_vector");
        assert!(m.columns == 1, "m.columns == 1 in matrix::c_add_vector"); // not really necessary, but useful for checking for proper use of method
        for row in 0..self.rows {
            for column in 0..self.columns {
                self.data[row][column] = self.data[row][column] + m.data[row][0];
            }
        }
    }
	
	//    pub fn inplace_c_add_vector(output: &mut Matrix, m: &Matrix) {
    pub fn c_add_vector(&mut self, m: &Matrix) {
        assert!(m.rows == self.rows, "m.rows == output.rows in matrix::c_add_vector");
        assert!(m.columns == 1, "m.columns == 1 in matrix::c_add_vector"); // not really necessary, but useful for checking for proper use of method
		for row in 0..self.rows {
			for column in 0..self.columns {
				self.data[row][column] += m.data[row][0];
			}
		}
    }
	
    //    pub fn c_apply_softmax(output: &mut Matrix, m: &Matrix) {
    pub fn c_apply_softmax2(&mut self, m: &Matrix) {
        let base: f32 = std::f32::consts::E;
        for col in 0..self.columns {
            let mut sum: f32 = 0.0;
            let mut exp_entry = vec![0.0; self.rows];

            for row in 0..m.rows {
                let input = base.powf(m.data[row][col]);
                sum += input;
                exp_entry[row] = input;            
            }
            
            for row in 0..self.rows {
                self.data[row][col] = exp_entry[row]/sum
            }
        }        
    }
	
	pub fn c_apply_softmax(&mut self, m: &Matrix) {
        let base: f32 = std::f32::consts::E;
		let mut max_holder = vec![0.0; m.columns];
		
        for row in 0..self.rows {
			for col in 0..self.columns {
				let input = base.powf(m.data[row][col]);
				self.data[row][col] = input;
				max_holder[col] += input;
			}
		}	
		for row in 0..self.rows {
			for col in 0..self.columns {
				self.data[row][col] /= max_holder[col];
			}
		}
    }
	

    //   pub fn inplace_e_apply_function(output: &mut Matrix, activation_fn: &dyn Fn(f32) -> f32) {    
    pub fn e_apply_function(&mut self, activation_fn: &dyn Fn(f32) -> f32) {
        for row in 0..self.rows {
            for column in 0..self.columns {
                self.data[row][column] = activation_fn(self.data[row][column]);
            }
        }
    }


    //    pub fn inplace_e_power(output: &mut Matrix, power: f32) {
    pub fn e_power_f32(&mut self, power: f32) {
        // TODO - delete, not used?
        for row in 0..self.rows {
            for column in 0..self.columns {
                self.data[row][column] = self.data[row][column].powf(power);
            }
        }
    }
	
	pub fn predict_from_softmax2(&mut self, m: &Matrix) {
        for col in 0..m.columns {
            let mut max: f32 = 0.0;
            let mut max_idx: f32 = 0.0;
            for row in 0..m.rows {
                if m.data[row][col] > max {
                    max = m.data[row][col];
                    max_idx = row as f32;
                }
            }
            self.data[0][col] = max_idx;
        }        
    }
	
	pub fn predict_from_softmax(&mut self, m: &Matrix) {
		assert!(self.columns==m.columns, "self.columns==m.columns in matrix::predict_from_softmax");
		let mut max_holder = vec![0.0; m.columns];
				
		for row in 0..m.rows {
			for column in 0..m.columns {
				if m.data[row][column] > max_holder[column] {
					max_holder[column] = m.data[row][column];
					self.data[0][column] = row as f32;
				}
			    //self.data[0][column] = self.data[0][column].max(m.data[row][column]);
			}
		}
    }

}


    pub fn multATB(m1: &Matrix, m2: &Matrix, output: &mut Matrix) {
        assert!(m1.rows==m2.rows, "m1.rows==m2.rows in matrix::multATB");
        assert!(m1.columns==output.rows, "m1.colums==output.rows in matrix::multATB");
        assert!(m2.columns==output.columns, "m2.colums==output.columns in matrix::multATB");

        let mut unordered_rows = (0..m1.columns)
            .into_par_iter()
            .map(move |i| {
            (i, (0..m2.columns)
                    .map(|j| (0..m1.rows).map(|k| m1.data[k][i] * m2.data[k][j]).sum())
                    .collect::<Vec<f32>>())
        })
        .collect::<Vec<(usize, Vec<f32>)>>();

        unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));
        output.data = unordered_rows.into_iter().map(|(_, row)| row).collect();
    }


    pub fn multABT(m1: &Matrix, m2: &Matrix, output: &mut Matrix) {
        assert!(m1.columns==m2.columns, "m1.columns==m2.columns in matrix::multABT");
        assert!(m1.rows==output.rows, "m1.rows==output.rows in matrix::multABT");
        assert!(m2.rows==output.columns, "m2.rows==output.columns in matrix::multABT");

        /*output.data = (0..m1.rows).
            map(|i|{
                (0..m2.rows)
                    .map(|j| (0..m1.columns).map(|k| m1.data[i][k] * m2.data[j][k]).sum())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();*/

        let mut unordered_rows = (0..m1.rows)
            .into_par_iter()
            .map(move |i| {
            (i, (0..m2.rows)
                    .map(|j| (0..m1.columns).map(|k| m1.data[i][k] * m2.data[j][k]).sum())
                    .collect::<Vec<f32>>())
        })
        .collect::<Vec<(usize, Vec<f32>)>>();

        unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));
        output.data = unordered_rows.into_iter().map(|(_, row)| row).collect();  
    }
 

    pub fn e_apply_function(output: &mut Matrix, m: &Matrix, activation_fn: &dyn Fn(f32) -> f32) {
        for row in 0..output.rows {
            for column in 0..output.columns {
                output.data[row][column] = activation_fn(m.data[row][column]);
            }
        }
    }


    pub fn e_multAB(output: &mut Matrix, m1: &Matrix, m2: &Matrix) {
        assert!(output.rows==m1.rows, "output.rows==m1.rows in matrix::e_multAB");
        assert!(output.rows==m2.rows, "output.rows==m2.rows in matrix::e_multAB");
        assert!(output.columns==m1.columns, "output.columns==m1.columns in matrix::e_multAB");
        assert!(output.columns==m2.columns, "output.columns==m2.columns in matrix::e_multAB");
        assert!(m1.rows==m2.rows, "m1.rows==m2.rows in matrix::e_multAB");
        assert!(m1.columns==m2.columns, "m1.columns==m2.columns in matrix::e_multAB");

        for row in 0..m2.rows {
            for column in 0..m2.columns {
                output.data[row][column] = m1.data[row][column] * m2.data[row][column];
            }
        }
    }

    
    pub fn bias_from_delta2(output: &mut Matrix, m: &Matrix) {
        assert!(output.rows==m.rows, "output.rows==m.rows in matrix::bias_from_delta");
        assert!(output.columns==1, "output.rows==m.rows in matrix::bias_from_delta");
        let unit = Matrix::ones_new(m.columns, 1);
        output.multAB(&m, &unit);
    }
	
	pub fn bias_from_delta(output: &mut Matrix, m: &Matrix) {
		assert!(output.rows==m.rows, "output.rows==m.rows in matrix::bias_from_delta");
		assert!(output.columns==1, "output.rows==m.rows in matrix::bias_from_delta");
		for row in 0..output.rows {
			output.data[row][0] = 0.0;
			for column in 0..output.columns {
				output.data[row][0] += m.data[row][column];				
			}
		}
	}

    pub fn get_softmax_delta2(output: &mut Matrix, m: &Matrix, t: &Matrix) {
        assert!(output.rows==m.rows, "output.rows==m.rows in matrix::get_softmax_delta");
        assert!(output.columns==m.columns, "output.columns==m.columns in matrix::get_softmax_delta");
        assert!(t.rows==1, "t.rows==1 in matrix::get_softmax_delta");
        assert!(t.columns==m.columns, "t.columns==m.columns in matrix::get_softmax_delta");
        assert!(t.columns==output.columns, "t.columns==m.columns in matrix::get_softmax_delta");
        
        for col in 0..output.columns {
            for row in 0..output.rows {
                if t.data[0][col] == row as f32 {
                output.data[row][col] = m.data[row][col] - 1.0;
                } else {
                    output.data[row][col] = m.data[row][col];
                }
            }
        }
	}
	
	pub fn get_softmax_delta(output: &mut Matrix, m: &Matrix, t: &Matrix) {
		assert!(output.rows==m.rows, "output.rows==m.rows in matrix::get_softmax_delta");
		assert!(output.columns==m.columns, "output.columns==m.columns in matrix::get_softmax_delta");
		assert!(t.rows==1, "t.rows==1 in matrix::get_softmax_delta");
		assert!(t.columns==m.columns, "t.columns==m.columns in matrix::get_softmax_delta");
		assert!(t.columns==output.columns, "t.columns==m.columns in matrix::get_softmax_delta");
		
		for row in 0..output.rows {
			for col in 0..output.columns {
				if t.data[0][col] == row as f32 {
				output.data[row][col] = m.data[row][col] - 1.0;
				} else {
					output.data[row][col] = m.data[row][col];
				}
			}
		}
	}
    
    

    



// =============== TESTS ===============
#[test]
fn test_another() {
    let a = Matrix::ones_new(2,3);
    let b = Matrix::ones_new(3,2);

    let c = Matrix::compute_matrix_combinators(&a, &b);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(c[i][j], 3.0);
        }
    }
}


#[test]
fn test_multAB() {
	let a = Matrix::ones_new(2,3);
	let b = Matrix::ones_new(3,2);
	let mut result = Matrix::new(2,2);
	result.multAB(&a, &b);

	for i in 0..2 {
		for j in 0..2 {
			assert_eq!(result.data[i][j], 3.0);
		}
	}
}


#[test]
fn test_transpose() {
	let mut a = Matrix::ones_new(2,3);
	a.data[0][0] = 0.0;
	a.data[0][1] = 1.0;
	a.data[0][2] = 2.0;
	a.data[1][0] = 10.0;
	a.data[1][1] = 11.0;
	a.data[1][2] = 12.0;
	
	let mut result = Matrix::new(3,2);
	a.transpose(&mut result);
	assert_eq!(result.data[0][0], 0.0);
    assert_eq!(result.data[1][0], 1.0);
    assert_eq!(result.data[2][0], 2.0);
    assert_eq!(result.data[0][1], 10.0);
    assert_eq!(result.data[1][1], 11.0);
    assert_eq!(result.data[2][1], 12.0);
}


#[test]
fn test_multATB() {
	let a = Matrix::ones_new(2,3);
	let b = Matrix::ones_new(2,3);
	let mut result = Matrix::new(3,3);

	multATB(&a, &b, & mut result);

	for i in 0..3 {
		for j in 0..3 {
			assert_eq!(result.data[i][j], 2.0);
		}
	}
}


#[test]
fn test_multABT() {
	let a = Matrix::ones_new(3,2);
	let b = Matrix::ones_new(3,2);
	let mut result = Matrix::new(3,3);

	multABT(&a, &b, & mut result);

	for i in 0..3 {
		for j in 0..3 {
			assert_eq!(result.data[i][j], 2.0);
		}
	}
}

#[test]
fn test_c_add_vector() {
	let mut a = Matrix::ones_new(2,3);
	a.data[0][0] = 0.0;
	a.data[0][1] = 1.0;
	a.data[0][2] = 2.0;
	a.data[1][0] = 10.0;
	a.data[1][1] = 11.0;
	a.data[1][2] = 12.0;
	
	let mut b = Matrix::new(2, 1);
	b.data[0][0] = 1.0;
	b.data[1][0] = 2.0;

	a.c_add_vector(& b);

	assert_eq!(a.data[0][0], 1.0);
	assert_eq!(a.data[0][1], 2.0);
	assert_eq!(a.data[0][2], 3.0);
	assert_eq!(a.data[1][0], 12.0);
	assert_eq!(a.data[1][1], 13.0);
	assert_eq!(a.data[1][2], 14.0);
}

#[test]
fn test_c_apply_softmax() {
    let mut a = Matrix::new(3,2);
	let mut result = Matrix::new(3,2);
	a.data[0][0] = 0.0;
	a.data[1][0] = 1.0;
	a.data[2][0] = 2.0;
	a.data[0][1] = 15.0;
	a.data[1][1] = 11.0;
	a.data[2][1] = 12.0;
	
	result.c_apply_softmax(&a);

	assert!(result.data[0][0] < result.data[1][0]);
	assert!(result.data[1][0] < result.data[2][0]);
	assert!(result.data[0][1] > result.data[2][1]);
	assert!(result.data[2][1] > result.data[1][1]);    

// 	assert_eq!(result.data[0][0], 0.0900306);
//        assert_eq!(result.data[1][0], 0.244728);
//	assert_eq!(result.data[2][0], 0.665241);
// 	assert_eq!(result.data[0][1], 0.93623955187651);
//	assert_eq!(result.data[1][1], 0.01714782554552);
//	assert_eq!(result.data[2][1], 0.046612622577974);
}

#[test]
fn test_e_multAB() {
    let mut a = Matrix::new(2,3);
	a.data[0][0] = 0.0;
	a.data[0][1] = 1.0;
	a.data[0][2] = 2.0;
	a.data[1][0] = 10.0;
	a.data[1][1] = 11.0;
	a.data[1][2] = 12.0;
	
	let mut b = Matrix::new(2,3);
	b.data[0][0] = 0.0;
	b.data[0][1] = 1.0;
	b.data[0][2] = 2.0;
	b.data[1][0] = 10.0;
	b.data[1][1] = 11.0;
	b.data[1][2] = 12.0;
	
	let mut result = Matrix::new(2,3);
	e_multAB(& mut result, & a, &b);

	assert_eq!(result.data[0][0], 0.0);
	assert_eq!(result.data[0][1], 1.0);
	assert_eq!(result.data[0][2], 4.0);
	assert_eq!(result.data[1][0], 100.0);
	assert_eq!(result.data[1][1], 11.0*11.0);
	assert_eq!(result.data[1][2], 12.0*12.0);
}


#[test]
fn test_e_mult_matrix() {
    let mut a = Matrix::new(2,3);
	a.data[0][0] = 0.0;
	a.data[0][1] = 1.0;
	a.data[0][2] = 2.0;
	a.data[1][0] = 10.0;
	a.data[1][1] = 11.0;
	a.data[1][2] = 12.0;
	
	let mut b = Matrix::new(2,3);
	b.data[0][0] = 0.0;
	b.data[0][1] = 1.0;
	b.data[0][2] = 2.0;
	b.data[1][0] = 10.0;
	b.data[1][1] = 11.0;
	b.data[1][2] = 12.0;

	a.e_mult_matrix(&b);

	assert_eq!(a.data[0][0], 0.0);
	assert_eq!(a.data[0][1], 1.0);
	assert_eq!(a.data[0][2], 4.0);
	assert_eq!(a.data[1][0], 100.0);
	assert_eq!(a.data[1][1], 11.0*11.0);
	assert_eq!(a.data[1][2], 12.0*12.0);
}

#[test]
fn test_e_mult_f32() {

    let mut output = Matrix::new(3, 3);
    output.data[0][0] = 5.0;
    output.data[1][0] = 20.0;
    output.data[2][0] = 300.0;
    output.data[0][1] = 1.0;
    output.data[1][1] = 0.02;
    output.data[2][1] = 0.7;
    output.data[0][2] = 3.0;
    output.data[1][2] = 3000.0;
    output.data[2][2] = 40.0;

    output.e_mult_f32(1.0/10.0);

    assert_eq!(output.data[0][0], 0.5);
    assert_eq!(output.data[1][0], 2.0);
    assert_eq!(output.data[2][0], 30.0);
    assert_eq!(output.data[0][1], 0.1);
    assert_eq!(output.data[1][1], 0.002);
    assert_eq!(output.data[2][1], 0.07);
    assert_eq!(output.data[0][2], 0.3);
    assert_eq!(output.data[1][2], 300.0);
    assert_eq!(output.data[2][2], 4.0);
}



#[test]
fn test_e_apply_function() {
	let mut a = Matrix::new(2,3);
	a.data[0][0] = 0.0;
	a.data[0][1] = -1.0;
	a.data[0][2] = -2.0;
	a.data[1][0] = -10.0;
	a.data[1][1] = -11.0;
	a.data[1][2] = -12.0;
	
	a.e_apply_function(&f32::abs);

	assert_eq!(a.data[0][0], 0.0);
	assert_eq!(a.data[0][1], 1.0);
	assert_eq!(a.data[0][2], 2.0);
	assert_eq!(a.data[1][0], 10.0);
	assert_eq!(a.data[1][1], 11.0);
	assert_eq!(a.data[1][2], 12.0);
}


#[test]
fn test_e_apply_function2() {
	let mut a = Matrix::new(2,3);
	let mut result = Matrix::new(2,3);
	a.data[0][0] = 0.0;
	a.data[0][1] = -1.0;
	a.data[0][2] = -2.0;
	a.data[1][0] = -10.0;
	a.data[1][1] = -11.0;
	a.data[1][2] = -12.0;
	
	e_apply_function(& mut result, &a, &f32::abs);

	assert_eq!(result.data[0][0], 0.0);
	assert_eq!(result.data[0][1], 1.0);
	assert_eq!(result.data[0][2], 2.0);
	assert_eq!(result.data[1][0], 10.0);
	assert_eq!(result.data[1][1], 11.0);
	assert_eq!(result.data[1][2], 12.0);
}


#[test]
fn test_predict_from_softmax() {
    let mut a = Matrix::new(3,2);
	let mut result = Matrix::new(1,2);
	a.data[0][0] = 0.1;
	a.data[1][0] = 0.2;
	a.data[2][0] = 0.7;
	a.data[0][1] = 0.15;
	a.data[1][1] = 0.6;
	a.data[2][1] = 0.25;
	
	result.predict_from_softmax2(&a);
	println!("result:\t{:?}\n", result);

	assert_eq!(result.data[0][0], 2.0);
	assert_eq!(result.data[0][1], 1.0);
	
	a.data[0][0] = 0.7;
	a.data[1][0] = 0.2;
	a.data[2][0] = 0.1;
	a.data[0][1] = 0.15;
	a.data[1][1] = 0.25;
	a.data[2][1] = 0.6;
	
	result.predict_from_softmax2(&a);
	println!("result:\t{:?}\n", result);
	
	assert_eq!(result.data[0][0], 0.0);
	assert_eq!(result.data[0][1], 2.0);
	
}



#[test]
fn test_get_softmax_delta() {
    let mut input = Matrix::new(3,3);
    input.data[0][0] = 0.5;
    input.data[1][0] = 0.2;
    input.data[2][0] = 0.3;
    input.data[0][1] = 0.1;
    input.data[1][1] = 0.2;
    input.data[2][1] = 0.7;
    input.data[0][2] = 0.3;
    input.data[1][2] = 0.3;
    input.data[2][2] = 0.4;

    let mut targets = Matrix::new(1,3);
    targets.data[0][0] = 0.0;
    targets.data[0][1] = 2.0;
    targets.data[0][2] = 1.0;

    let mut output = Matrix::new(3,3);
    get_softmax_delta(&mut output, &input, &targets);
    
    assert_eq!(output.data[0][0], -0.5);
    assert_eq!(output.data[1][0], 0.2);
    assert_eq!(output.data[2][0], 0.3);
    assert_eq!(output.data[0][1], 0.1);
    assert_eq!(output.data[1][1], 0.2);
    assert_eq!(output.data[2][1], -0.3);
    assert_eq!(output.data[0][2], 0.3);
    assert_eq!(output.data[1][2], -0.7);
    assert_eq!(output.data[2][2], 0.4);
        
}
