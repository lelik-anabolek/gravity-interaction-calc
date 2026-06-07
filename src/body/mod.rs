pub mod body {

    use nalgebra::Vector3;
    use rand::{rng, RngExt};
    use std::f64::consts::PI;
    #[derive(Debug, Clone)]
    pub struct Body {
        pub position: Vector3<f64>,
        pub velocity: Vector3<f64>,
        pub mass: f64,
        pub radius: f64,
    }

    impl Body {
        pub fn new(
            x: f64,
            y: f64,
            z: f64,
            dot_x: f64,
            dot_y: f64,
            dot_z: f64,
            mass: f64,
            radius: f64,
        ) -> Self {
            Self {
                position: Vector3::new(x, y, z),
                velocity: Vector3::new(dot_x, dot_y, dot_z),
                mass,
                radius,
            }
        }

        pub fn gen_random() -> Self {
            let mut rng = rng();

            let mass: f64 = rng.random_range(0.1..=100.0);
            let (pos_x, pos_y, pos_z): (f64, f64, f64) = (
                rng.random_range(-10.0..=10.0),
                rng.random_range(-10.0..=10.0),
                rng.random_range(-10.0..=10.0),
            );

            let velocity: f64 = rng.random_range(0.0..=1.0);
            let (theta, sigma): (f64, f64) = (
                rng.random_range(0.0..=2.0 * PI),
                rng.random_range(0.0..=2.0 * PI),
            );

            let dot_x = velocity * theta.sin() * sigma.cos();
            let dot_y = velocity * theta.sin() * sigma.sin();
            let dot_z = velocity * theta.cos();

            let radius = rng.random_range(1.0..=10.0);

            Self::new(pos_x, pos_y, pos_z, dot_x, dot_y, dot_z, mass, radius)
        }
    }
}
