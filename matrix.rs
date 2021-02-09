use rand::distributions::{Distribution, Normal};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<Vec<f32>>,
}
impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        // create new matrix of 0s
        Matrix {
            rows: rows,
            columns: columns,
            data: vec![vec![0.0; columns]; rows],
        }
    }

    pub fn randn_new(rows: usize, columns: usize, mean: f32, var: f32) -> Matrix {
        // initiliazes value from N(mean, var)
        let normal = Normal::new(mean as f64, 1.0);
        let mut data = Vec::new();
        for _r in 0..rows {
            let mut row_data = Vec::new();
            for _c in 0..columns {
                row_data.push((normal.sample(&mut rand::thread_rng()) as f32) * var);
            }
            data.push(row_data);
        }

        Matrix {
            rows: rows,
            columns: columns,
            data: data,
        }
    }

    pub fn multAB(&mut self, m1: &Matrix, m2: &Matrix) {
        // multiply two matrices A * B
        assert!(
            m1.columns == m2.rows,
            "m1.columns==m2.rows in matrix::multAB"
        );
        assert!(m1.rows == self.rows, "m1.rows==self.rows in matrix::multAB");
        assert!(
            m2.columns == self.columns,
            "m2.columns==self.columns in matrix::multAB"
        );

        let mut unordered_rows = (0..m1.rows)
            .into_par_iter()
            .map(move |i| {
                let m1_row = &m1.data[i];

                (
                    i,
                    (0..m2.columns)
                        .map(|j| (0..m2.rows).map(|k| m1_row[k] * m2.data[k][j]).sum())
                        .collect::<Vec<f32>>(),
                )
            })
            .collect::<Vec<(usize, Vec<f32>)>>();

        unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));
        self.data = unordered_rows.into_iter().map(|(_, row)| row).collect();
    }

    pub fn c_add_vector(&mut self, m: &Matrix) {
        // add vector to a matrix columnwise
        assert!(
            m.rows == self.rows,
            "m.rows == output.rows in matrix::c_add_vector"
        );
        assert!(m.columns == 1, "m.columns == 1 in matrix::c_add_vector"); // not really necessary, but useful for checking for proper use of method
        for row in 0..self.rows {
            for column in 0..self.columns {
                self.data[row][column] += m.data[row][0];
            }
        }
    }

    pub fn c_apply_softmax(&mut self, m: &Matrix) {
        // apply softmax columnwise
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

    pub fn predict_from_softmax(&mut self, m: &Matrix) {
        assert!(
            self.columns == m.columns,
            "self.columns==m.columns in matrix::predict_from_softmax"
        );
        let mut max_holder = vec![0.0; m.columns];

        for row in 0..m.rows {
            for column in 0..m.columns {
                if m.data[row][column] > max_holder[column] {
                    max_holder[column] = m.data[row][column];
                    self.data[0][column] = row as f32;
                }
            }
        }
    }
}

pub fn multATB(m1: &Matrix, m2: &Matrix, output: &mut Matrix) {
    // multiply two matrices, transposed(A) * B
    assert!(m1.rows == m2.rows, "m1.rows==m2.rows in matrix::multATB");
    assert!(
        m1.columns == output.rows,
        "m1.colums==output.rows in matrix::multATB"
    );
    assert!(
        m2.columns == output.columns,
        "m2.colums==output.columns in matrix::multATB"
    );

    let mut unordered_rows = (0..m1.columns)
        .into_par_iter()
        .map(move |i| {
            (
                i,
                (0..m2.columns)
                    .map(|j| (0..m1.rows).map(|k| m1.data[k][i] * m2.data[k][j]).sum())
                    .collect::<Vec<f32>>(),
            )
        })
        .collect::<Vec<(usize, Vec<f32>)>>();

    unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));
    output.data = unordered_rows.into_iter().map(|(_, row)| row).collect();
}

pub fn multABT(m1: &Matrix, m2: &Matrix, output: &mut Matrix) {
    // multiply two matrices, A * transposed(B)
    assert!(
        m1.columns == m2.columns,
        "m1.columns==m2.columns in matrix::multABT"
    );
    assert!(
        m1.rows == output.rows,
        "m1.rows==output.rows in matrix::multABT"
    );
    assert!(
        m2.rows == output.columns,
        "m2.rows==output.columns in matrix::multABT"
    );

    let mut unordered_rows = (0..m1.rows)
        .into_par_iter()
        .map(move |i| {
            (
                i,
                (0..m2.rows)
                    .map(|j| (0..m1.columns).map(|k| m1.data[i][k] * m2.data[j][k]).sum())
                    .collect::<Vec<f32>>(),
            )
        })
        .collect::<Vec<(usize, Vec<f32>)>>();

    unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));
    output.data = unordered_rows.into_iter().map(|(_, row)| row).collect();
}

pub fn e_apply_function(output: &mut Matrix, m: &Matrix, activation_fn: &dyn Fn(f32) -> f32) {
    // apply function elemntwise to the matrix m, result save into output matrix
    for row in 0..output.rows {
        for column in 0..output.columns {
            output.data[row][column] = activation_fn(m.data[row][column]);
        }
    }
}

pub fn e_multAB(output: &mut Matrix, m1: &Matrix, m2: &Matrix) {
    // multiply two matrices elementwise, result save into output matrix
    assert!(
        output.rows == m1.rows,
        "output.rows==m1.rows in matrix::e_multAB"
    );
    assert!(
        output.rows == m2.rows,
        "output.rows==m2.rows in matrix::e_multAB"
    );
    assert!(
        output.columns == m1.columns,
        "output.columns==m1.columns in matrix::e_multAB"
    );
    assert!(
        output.columns == m2.columns,
        "output.columns==m2.columns in matrix::e_multAB"
    );
    assert!(m1.rows == m2.rows, "m1.rows==m2.rows in matrix::e_multAB");
    assert!(
        m1.columns == m2.columns,
        "m1.columns==m2.columns in matrix::e_multAB"
    );

    for row in 0..m2.rows {
        for column in 0..m2.columns {
            output.data[row][column] = m1.data[row][column] * m2.data[row][column];
        }
    }
}

pub fn bias_from_delta(output: &mut Matrix, m: &Matrix) {
    // get bias from deltamatrix m, result save in output
    assert!(
        output.rows == m.rows,
        "output.rows==m.rows in matrix::bias_from_delta"
    );
    assert!(
        output.columns == 1,
        "output.rows==m.rows in matrix::bias_from_delta"
    );
    for row in 0..output.rows {
        output.data[row][0] = 0.0;
        for column in 0..output.columns {
            output.data[row][0] += m.data[row][column];
        }
    }
}

pub fn get_softmax_delta(output: &mut Matrix, m: &Matrix, t: &Matrix) {
    // get delta from softmax predictions
    assert!(
        output.rows == m.rows,
        "output.rows==m.rows in matrix::get_softmax_delta"
    );
    assert!(
        output.columns == m.columns,
        "output.columns==m.columns in matrix::get_softmax_delta"
    );
    assert!(t.rows == 1, "t.rows==1 in matrix::get_softmax_delta");
    assert!(
        t.columns == m.columns,
        "t.columns==m.columns in matrix::get_softmax_delta"
    );
    assert!(
        t.columns == output.columns,
        "t.columns==m.columns in matrix::get_softmax_delta"
    );

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
fn test_c_add_vector() {
    let mut a = Matrix::new(2, 3);
    a.data[0][0] = 0.0;
    a.data[0][1] = 1.0;
    a.data[0][2] = 2.0;
    a.data[1][0] = 10.0;
    a.data[1][1] = 11.0;
    a.data[1][2] = 12.0;

    let mut b = Matrix::new(2, 1);
    b.data[0][0] = 1.0;
    b.data[1][0] = 2.0;

    a.c_add_vector(&b);

    assert_eq!(a.data[0][0], 1.0);
    assert_eq!(a.data[0][1], 2.0);
    assert_eq!(a.data[0][2], 3.0);
    assert_eq!(a.data[1][0], 12.0);
    assert_eq!(a.data[1][1], 13.0);
    assert_eq!(a.data[1][2], 14.0);
}

#[test]
fn test_c_apply_softmax() {
    let mut a = Matrix::new(3, 2);
    let mut result = Matrix::new(3, 2);
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
}

#[test]
fn test_e_multAB() {
    let mut a = Matrix::new(2, 3);
    a.data[0][0] = 0.0;
    a.data[0][1] = 1.0;
    a.data[0][2] = 2.0;
    a.data[1][0] = 10.0;
    a.data[1][1] = 11.0;
    a.data[1][2] = 12.0;

    let mut b = Matrix::new(2, 3);
    b.data[0][0] = 0.0;
    b.data[0][1] = 1.0;
    b.data[0][2] = 2.0;
    b.data[1][0] = 10.0;
    b.data[1][1] = 11.0;
    b.data[1][2] = 12.0;

    let mut result = Matrix::new(2, 3);
    e_multAB(&mut result, &a, &b);

    assert_eq!(result.data[0][0], 0.0);
    assert_eq!(result.data[0][1], 1.0);
    assert_eq!(result.data[0][2], 4.0);
    assert_eq!(result.data[1][0], 100.0);
    assert_eq!(result.data[1][1], 11.0 * 11.0);
    assert_eq!(result.data[1][2], 12.0 * 12.0);
}

#[test]
fn test_get_softmax_delta() {
    let mut input = Matrix::new(3, 3);
    input.data[0][0] = 0.5;
    input.data[1][0] = 0.2;
    input.data[2][0] = 0.3;
    input.data[0][1] = 0.1;
    input.data[1][1] = 0.2;
    input.data[2][1] = 0.7;
    input.data[0][2] = 0.3;
    input.data[1][2] = 0.3;
    input.data[2][2] = 0.4;

    let mut targets = Matrix::new(1, 3);
    targets.data[0][0] = 0.0;
    targets.data[0][1] = 2.0;
    targets.data[0][2] = 1.0;

    let mut output = Matrix::new(3, 3);
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
