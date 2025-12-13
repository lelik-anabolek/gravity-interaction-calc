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

        let diff_x = (&x * ones.clone()) - (ones_transposed * x.transpose());
        let diff_y = (&y * ones.clone()) - (ones_transposed * y.transpose());
        let diff_z = (&z * ones.clone()) - (ones_transposed * z.transpose());

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
mod test {
    mod center_of_mass {
        use crate::body::body::Body;
        use crate::trajectory::trajectory::calc_center_of_mass;

        #[test]
        fn test_center_of_mass_basic() {
            let bodies = [
                Body::new(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                Body::new(-1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                Body::new(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0),
            ];

            let com = calc_center_of_mass(&bodies);

            // Waiting:
            // (1*1 + -1*1 + 0*2) / 4 = 0
            // (0*1 + 0*1 + 1*2) / 4 = 0.5
            // z = 0
            assert!((com.x - 0.0).abs() < 1e-12);
            assert!((com.y - 0.5).abs() < 1e-12);
            assert!((com.z - 0.0).abs() < 1e-12);
        }

        #[test]
        fn test_center_of_mass_symmetric() {
            let bodies = [
                Body::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
                Body::new(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
                Body::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
            ];

            let com = calc_center_of_mass(&bodies);

            assert!((com.x - 0.0).abs() < 1e-12);
            assert!((com.y - 0.0).abs() < 1e-12);
            assert!((com.z - (1.0 / 3.0)).abs() < 1e-12);
        }
        #[test]
        fn test_center_of_mass_different_masses() {
            let bodies = [
                Body::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 1.0),
                Body::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
                Body::new(0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
            ];

            let com = calc_center_of_mass(&bodies);

            // (0*10 + 10*1 + 0*1) / 12 = 10 / 12 ≈ 0.8333
            // (0*10 + 0*1 + 10*1) / 12 = 10 / 12 ≈ 0.8333
            // z = 0
            assert!((com.x - (10.0 / 12.0)).abs() < 1e-12);
            assert!((com.y - (10.0 / 12.0)).abs() < 1e-12);
            assert!((com.z - 0.0).abs() < 1e-12);
        }
        #[test]
        fn test_center_of_mass_all_same_point() {
            let bodies = [
                Body::new(2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0),
                Body::new(2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 5.0, 1.0),
                Body::new(2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.0),
            ];

            let com = calc_center_of_mass(&bodies);

            assert!((com.x - 2.0).abs() < 1e-12);
            assert!((com.y - 2.0).abs() < 1e-12);
            assert!((com.z - 2.0).abs() < 1e-12);
        }
    }

    mod has_collision {
        use crate::body::body::Body;
        use crate::trajectory::trajectory::has_collision;
        use nalgebra::Vector3;

        fn body_at(x: f64, y: f64, z: f64, r: f64) -> Body {
            Body {
                position: Vector3::new(x, y, z),
                velocity: Vector3::zeros(),
                mass: 1.0,
                radius: r,
            }
        }

        #[test]
        fn test_no_collisions() {
            let bodies = [
                body_at(0.0, 0.0, 0.0, 1.0),
                body_at(5.0, 0.0, 0.0, 1.0),
                body_at(0.0, 5.0, 0.0, 1.0),
            ];

            assert_eq!(has_collision(&bodies), false);
        }

        #[test]
        fn test_touching_collision() {
            let bodies = [
                body_at(0.0, 0.0, 0.0, 1.0),
                body_at(2.0, 0.0, 0.0, 1.0),
                body_at(10.0, 0.0, 0.0, 1.0),
            ];

            assert_eq!(has_collision(&bodies), true);
        }

        #[test]
        fn test_deep_overlap_collision() {
            let bodies = [
                body_at(0.0, 0.0, 0.0, 1.0),
                body_at(1.0, 0.0, 0.0, 1.0),
                body_at(100.0, 0.0, 0.0, 1.0),
            ];

            assert_eq!(has_collision(&bodies), true);
        }

        #[test]
        fn test_only_some_collide() {
            let bodies = [
                body_at(0.0, 0.0, 0.0, 1.0),
                body_at(0.0, 0.0, 10.0, 2.0),
                body_at(0.0, 0.0, 12.0, 2.0),
            ];

            assert_eq!(has_collision(&bodies), true);
        }

        #[test]
        fn test_all_collide() {
            let bodies = [
                body_at(0.0, 0.0, 0.0, 5.0),
                body_at(3.0, 0.0, 0.0, 5.0),
                body_at(-3.0, 0.0, 0.0, 5.0),
            ];

            assert_eq!(has_collision(&bodies), true);
        }
        #[test]
        fn test_exact_tangency_collision() {
            let bodies = [
                body_at(0.0, 0.0, 0.0, 1.0),
                body_at(2.0, 0.0, 0.0, 1.0),
                body_at(100.0, 0.0, 0.0, 1.0),
            ];
            assert_eq!(has_collision(&bodies), true);
        }
    }

    mod calc_gravity_acceleration {

        use crate::trajectory::trajectory::calc_gravity_acceleration;
        use nalgebra::{Matrix3, RowVector3};

        #[test]
        fn test_no_nan_and_no_inf_excluding_diagonal() {
            let position = Matrix3::new(0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0);
            let mass_mat = Matrix3::from_element(1.0);

            let acc = calc_gravity_acceleration(&position, &mass_mat);

            for i in 0..3 {
                for j in 0..3 {
                    assert!(!acc[(i, j)].is_nan());
                    assert!(!acc[(i, j)].is_infinite());
                }
            }
        }

        #[test]
        fn test_gravity_symmetric() {
            let position = Matrix3::from_rows(&[
                RowVector3::new(-1.0, 0.0, 1.0), // x
                RowVector3::new(0.0, 0.0, 0.0),  // y
                RowVector3::new(0.0, 0.0, 0.0),  // z
            ]);

            let mass_mat = Matrix3::from_rows(&[
                RowVector3::new(1.0, 1.0, 1.0), // x
                RowVector3::new(1.0, 1.0, 1.0), // y
                RowVector3::new(1.0, 1.0, 1.0), // z
            ]);

            let acc = calc_gravity_acceleration(&position, &mass_mat);

            assert!((acc[(0, 0)] - 1.25).abs() < 1e-6);

            // body 1:
            // right: 1/1^2
            // left: 1/1^2
            // sum = 0 (symmetry)
            assert!(acc[(0, 1)].abs() < 1e-12);

            // body 2:
            // to left: 1 + 0.25 = 1.25
            assert!((acc[(0, 2)] + 1.25).abs() < 1e-6);

            // y,z must be zero
            for axis in 1..3 {
                for i in 0..3 {
                    assert!(acc[(axis, i)].abs() < 1e-12);
                }
            }
        }

        #[test]
        fn test_perpendicular_bodies() {
            // located by axis X, Y и Z
            let position = Matrix3::from_rows(&[
                RowVector3::new(1.0, 0.0, 0.0), // x
                RowVector3::new(0.0, 1.0, 0.0), // y
                RowVector3::new(0.0, 0.0, 1.0), // z
            ]);
            let mass_mat = Matrix3::from_element(1.0);

            let acc = calc_gravity_acceleration(&position, &mass_mat);

            // distances √2 и √2 → vectors symmetrical directed
            // Check, that vector of acceleration is non-zero
            for i in 0..3 {
                let ax = acc[(0, i)];
                let ay = acc[(1, i)];
                let az = acc[(2, i)];
                let magnitude = (ax * ax + ay * ay + az * az).sqrt();
                assert!(magnitude > 0.0);
            }
        }
    }
}
