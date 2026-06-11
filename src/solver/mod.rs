use crate::body::Body;
use crate::trajectory::{calc_gravity_acceleration, has_collision};
use nalgebra::{DVector, Matrix3xX, MatrixXx1, RowDVector};
use ode_solvers::dop_shared::IntegrationError;
use ode_solvers::{Dopri5, System};

/*
State is one flat vector of length 6N: first every body's position,
then every body's velocity.

  [ x1 y1 z1  x2 y2 z2  ...  xN yN zN | vx1 vy1 vz1  ...  vxN vyN vzN ]
    \------------ positions ---------/ \------------ velocities -----/
              [0 .. 3N)                          [3N .. 6N)

Each body owns one (x, y, z) slot inside each block.
*/
type State = DVector<f64>;

/// Start index of body `i`'s position (x, y, z) inside the state vector.
fn calc_skipper_pos(i: usize) -> usize {
    i * 3
}

/// Start index of body `i`'s velocity (vx, vy, vz) inside the state vector.
/// `n` is the body count: the velocity block begins after all 3N positions.
fn calc_skipper_vel(i: usize, n: usize) -> usize {
    3 * n + i * 3
}

/// Gather every body's position from the flat state into a 3xN matrix
/// (column `i` == body `i`) — the shape `calc_gravity_acceleration` expects.
fn read_positions(y: &State, n: usize) -> Matrix3xX<f64> {
    let mut positions = Matrix3xX::zeros(n);

    for i in 0..n {
        let body_pos = y.fixed_rows::<3>(calc_skipper_pos(i));
        positions.column_mut(i).copy_from(&body_pos);
    }

    positions
}

struct ThreeBodySystem {
    pub masses: RowDVector<f64>,
    pub bodies: Vec<Body>,
}

impl ThreeBodySystem {
    pub fn init(bodies: &[Body]) -> Self {
        let mut mass_vec = RowDVector::<f64>::zeros(bodies.len());

        for (i, b) in bodies.iter().enumerate() {
            mass_vec[i] = b.mass;
        }

        Self {
            masses: mass_vec,
            bodies: bodies.to_vec(),
        }
    }
}

impl System<f64, State> for ThreeBodySystem {
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        let n = self.masses.len();

        let positions = read_positions(y, n);
        let acceleration = calc_gravity_acceleration(&positions, &self.masses);

        // Equations of motion, body by body:
        //   d(position)/dt = velocity
        //   d(velocity)/dt = acceleration
        for i in 0..n {
            let pos_slot = calc_skipper_pos(i);
            let vel_slot = calc_skipper_vel(i, n);

            let velocity = y.fixed_rows::<3>(vel_slot);
            let acc = acceleration.column(i);

            dy.fixed_rows_mut::<3>(pos_slot).copy_from(&velocity);
            dy.fixed_rows_mut::<3>(vel_slot).copy_from(&acc);
        }
    }

    fn solout(&mut self, _t: f64, y: &State, _dy: &State) -> bool {
        // Snapshot the integrated positions back into the bodies, then check for a crash.
        for (i, body) in self.bodies.iter_mut().enumerate() {
            body.position = y.fixed_rows::<3>(calc_skipper_pos(i)).into_owned();
        }

        has_collision(&self.bodies)
    }
}

fn init_state(bodies: &[Body]) -> State {
    let n = bodies.len();
    let mut state = State::zeros(6 * n);

    for (i, b) in bodies.iter().enumerate() {
        state
            .fixed_rows_mut::<3>(calc_skipper_pos(i))
            .copy_from(&b.position);
        state
            .fixed_rows_mut::<3>(calc_skipper_vel(i, n))
            .copy_from(&b.velocity);
    }

    state
}

#[derive(Debug)]
pub struct IntegrationData {
    x: Vec<f64>,
    y: Vec<MatrixXx1<f64>>,
}

pub fn integrate(
    bodies: &[Body],
    time: Option<f64>,
    step: Option<f64>,
) -> Result<IntegrationData, IntegrationError> {
    let system = ThreeBodySystem::init(bodies);
    let state = init_state(bodies);
    let mut stepper: Dopri5<f64, State, ThreeBodySystem> = Dopri5::new(
        system,
        0.0,
        time.unwrap_or(150.0),
        step.unwrap_or(0.002),
        state,
        1.0e-14,
        1.0e-14,
    );

    let res = stepper.integrate();

    match res {
        Ok(stats) => {
            println!("stats: {}", stats);
            Ok(IntegrationData {
                x: stepper.x_out().clone(),
                y: stepper.y_out().clone(),
            })
        }
        Err(e) => {
            return Err(e);
        }
    }
}
