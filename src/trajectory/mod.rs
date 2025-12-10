pub mod trajectory {
    use crate::body::body::Body;
    use nalgebra::{Matrix3, RowVector3, Vector3};
    const G: f64 = 1.0;

    pub fn calc_center_of_mass(bodies: &[Body; 3]) -> Vector3<f64> {
        let mut weighted_sum = Vector3::zeros();
        let mut total_mass = 0.0;

        for b in bodies {
            weighted_sum += b.position * b.mass;
            total_mass += b.mass;
        }

        weighted_sum / total_mass
    }

    pub fn has_collision(bodies: &[Body; 3]) -> bool {
        let n = bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = (bodies[i].position - bodies[j].position).norm();
                if dist <= bodies[i].radius + bodies[j].radius {
                    return true;
                }
            }
        }
        false
    }

    pub fn calc_gravity_acceleration(
        position: &Matrix3<f64>,
        mass_mat: &Matrix3<f64>,
    ) -> Matrix3<f64> {
        let n = 3;

        if n > 3 {
            panic!("Possible to calculate only 3 body");
        }

        let x = position.row(0).transpose();
        let y = position.row(1).transpose();
        let z = position.row(2).transpose();

        let ones = RowVector3::from_element(1.0);
        let ones_transposed = ones.transpose();
        let diff_x = (ones_transposed * x.transpose()) - (&x * ones.clone());
        let diff_y = (ones_transposed * y.transpose()) - (&y * ones.clone());
        let diff_z = (ones_transposed * z.transpose()) - (&z * ones.clone());

        let dist_sq = diff_x.component_mul(&diff_x)
            + diff_y.component_mul(&diff_y)
            + diff_z.component_mul(&diff_z);

        let mut dist_cube = dist_sq.map(|v| (v + 1e-12).powf(1.5));

        // ignore diagonal
        for i in 0..n {
            dist_cube[(i, i)] = f64::INFINITY;
        }

        let factor = mass_mat.component_div(&dist_cube);

        let acceleration_x = diff_x.component_mul(&factor).row_sum() * G;
        let acceleration_y = diff_y.component_mul(&factor).row_sum() * G;
        let acceleration_z = diff_z.component_mul(&factor).row_sum() * G;

        let mut acc = Matrix3::<f64>::zeros();

        acc.set_row(0, &acceleration_x);
        acc.set_row(1, &acceleration_y);
        acc.set_row(2, &acceleration_z);

        acc
    }
}

#[cfg(test)]
mod trajectory_test {
    use super::*;
    use crate::body::body::Body;
    use nalgebra::Vector3;

    #[test]
    fn center_of_mass() {
        let bodies = [
            Body::new(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            Body::new(-1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            Body::new(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0),
        ];

        assert_eq!(
            trajectory::calc_center_of_mass(&bodies),
            Vector3::new(0.0, 0.5, 0.0)
        );
    }
}
