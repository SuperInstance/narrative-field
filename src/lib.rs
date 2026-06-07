//! # Narrative Field
//!
//! Narrative structure modeled as vector field operations.
//! Stories are treated as fields where plot points are positions,
//! arcs are trajectories, tension is potential energy, resolution
//! is a stable fixed point, and narrator is a perspective transformation.

use std::f64::consts::PI;

// ── plot_point ──────────────────────────────────────────────────────────────

/// A position in narrative space defined by coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl PlotPoint {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn distance_to(&self, other: &PlotPoint) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }

    pub fn midpoint(&self, other: &PlotPoint) -> PlotPoint {
        PlotPoint::new(
            (self.x + other.x) / 2.0,
            (self.y + other.y) / 2.0,
            (self.z + other.z) / 2.0,
        )
    }

    pub fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    pub fn normalize(&self) -> PlotPoint {
        let m = self.magnitude();
        if m == 0.0 {
            PlotPoint::origin()
        } else {
            PlotPoint::new(self.x / m, self.y / m, self.z / m)
        }
    }

    pub fn scale(&self, factor: f64) -> PlotPoint {
        PlotPoint::new(self.x * factor, self.y * factor, self.z * factor)
    }

    pub fn add(&self, other: &PlotPoint) -> PlotPoint {
        PlotPoint::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn sub(&self, other: &PlotPoint) -> PlotPoint {
        PlotPoint::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn dot(&self, other: &PlotPoint) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &PlotPoint) -> PlotPoint {
        PlotPoint::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

// ── arc ─────────────────────────────────────────────────────────────────────

/// A trajectory through narrative states, defined by start, end, and curvature.
#[derive(Debug, Clone)]
pub struct Arc {
    pub start: PlotPoint,
    pub end: PlotPoint,
    pub curvature: f64,
}

impl Arc {
    pub fn new(start: PlotPoint, end: PlotPoint, curvature: f64) -> Self {
        Self { start, end, curvature }
    }

    pub fn linear(start: PlotPoint, end: PlotPoint) -> Self {
        Self { start, end, curvature: 0.0 }
    }

    pub fn length(&self) -> f64 {
        let d = self.start.distance_to(&self.end);
        if self.curvature == 0.0 {
            d
        } else {
            let theta = self.curvature.abs().min(PI);
            d * theta / (2.0 * (theta / 2.0).sin())
        }
    }

    pub fn interpolate(&self, t: f64) -> PlotPoint {
        let t = t.clamp(0.0, 1.0);
        let linear = PlotPoint::new(
            self.start.x + t * (self.end.x - self.start.x),
            self.start.y + t * (self.end.y - self.start.y),
            self.start.z + t * (self.end.z - self.start.z),
        );
        if self.curvature == 0.0 {
            linear
        } else {
            let offset = self.curvature * (PI * t).sin();
            let _mid = self.start.midpoint(&self.end);
            let perp = self.start.cross(&self.end).normalize().scale(offset);
            linear.add(&perp)
        }
    }

    pub fn reverse(&self) -> Arc {
        Arc::new(self.end.clone(), self.start.clone(), -self.curvature)
    }

    pub fn is_degenerate(&self) -> bool {
        self.start.distance_to(&self.end) < 1e-10
    }

    pub fn midpoint(&self) -> PlotPoint {
        self.interpolate(0.5)
    }

    pub fn split_at(&self, t: f64) -> (Arc, Arc) {
        let mid = self.interpolate(t);
        let half_curv = self.curvature / 2.0;
        (
            Arc::new(self.start.clone(), mid.clone(), half_curv),
            Arc::new(mid, self.end.clone(), half_curv),
        )
    }
}

// ── tension ─────────────────────────────────────────────────────────────────

/// Tension as potential energy of conflict in narrative space.
#[derive(Debug, Clone)]
pub struct Tension {
    pub magnitude: f64,
    pub direction: PlotPoint,
    pub decay_rate: f64,
}

impl Tension {
    pub fn new(magnitude: f64, direction: PlotPoint, decay_rate: f64) -> Self {
        Self { magnitude, direction: direction.normalize(), decay_rate }
    }

    pub fn zero() -> Self {
        Self { magnitude: 0.0, direction: PlotPoint::origin(), decay_rate: 0.0 }
    }

    pub fn potential_energy(&self) -> f64 {
        self.magnitude.powi(2) / 2.0
    }

    pub fn decay(&self, dt: f64) -> Tension {
        Tension {
            magnitude: self.magnitude * (-self.decay_rate * dt).exp(),
            direction: self.direction.clone(),
            decay_rate: self.decay_rate,
        }
    }

    pub fn is_resolved(&self, threshold: f64) -> bool {
        self.magnitude < threshold
    }

    pub fn add(&self, other: &Tension) -> Tension {
        let combined_dir = self.direction.scale(self.magnitude).add(&other.direction.scale(other.magnitude));
        let combined_mag = combined_dir.magnitude();
        Tension {
            magnitude: combined_mag,
            direction: combined_dir.normalize(),
            decay_rate: (self.decay_rate + other.decay_rate) / 2.0,
        }
    }

    pub fn escalate(&self, factor: f64) -> Tension {
        Tension {
            magnitude: self.magnitude * factor,
            direction: self.direction.clone(),
            decay_rate: self.decay_rate,
        }
    }

    pub fn redirect(&self, new_direction: PlotPoint) -> Tension {
        Tension {
            magnitude: self.magnitude,
            direction: new_direction.normalize(),
            decay_rate: self.decay_rate,
        }
    }

    pub fn force_at(&self, point: &PlotPoint) -> PlotPoint {
        let dist = point.distance_to(&PlotPoint::origin());
        if dist < 1e-10 {
            return self.direction.scale(self.magnitude);
        }
        let falloff = 1.0 / (1.0 + dist);
        self.direction.scale(self.magnitude * falloff)
    }
}

// ── resolution ──────────────────────────────────────────────────────────────

/// A stable fixed point where narrative tensions converge.
#[derive(Debug, Clone)]
pub struct Resolution {
    pub fixed_point: PlotPoint,
    pub basin_radius: f64,
    pub convergence_rate: f64,
}

impl Resolution {
    pub fn new(fixed_point: PlotPoint, basin_radius: f64, convergence_rate: f64) -> Self {
        Self { fixed_point, basin_radius, convergence_rate }
    }

    pub fn is_in_basin(&self, point: &PlotPoint) -> bool {
        point.distance_to(&self.fixed_point) <= self.basin_radius
    }

    pub fn converge_step(&self, point: &PlotPoint) -> PlotPoint {
        let diff = self.fixed_point.sub(point);
        point.add(&diff.scale(self.convergence_rate))
    }

    pub fn resolve_tension(&self, tension: &Tension, steps: usize) -> Tension {
        let mut t = tension.clone();
        for _ in 0..steps {
            t = t.decay(self.convergence_rate);
        }
        t
    }

    pub fn time_to_resolve(&self, tension: &Tension, threshold: f64) -> f64 {
        if tension.magnitude < threshold || tension.decay_rate <= 0.0 {
            return 0.0;
        }
        ((threshold / tension.magnitude).ln() / (-tension.decay_rate)).max(0.0)
    }

    pub fn stability(&self) -> f64 {
        1.0 / (1.0 + self.convergence_rate)
    }

    pub fn merge(&self, other: &Resolution) -> Resolution {
        let merged_point = self.fixed_point.midpoint(&other.fixed_point);
        Resolution {
            fixed_point: merged_point,
            basin_radius: (self.basin_radius + other.basin_radius) / 2.0,
            convergence_rate: (self.convergence_rate + other.convergence_rate) / 2.0,
        }
    }

    pub fn trajectory(&self, start: &PlotPoint, steps: usize) -> Vec<PlotPoint> {
        let mut points = vec![start.clone()];
        let mut current = start.clone();
        for _ in 0..steps {
            current = self.converge_step(&current);
            points.push(current.clone());
        }
        points
    }
}

// ── narrator ────────────────────────────────────────────────────────────────

/// A perspective transformation on narrative space.
#[derive(Debug, Clone)]
pub struct Narrator {
    pub position: PlotPoint,
    pub orientation: f64, // radians
    pub bias: f64,
}

impl Narrator {
    pub fn new(position: PlotPoint, orientation: f64, bias: f64) -> Self {
        Self { position, orientation, bias }
    }

    pub fn omniscient() -> Self {
        Self { position: PlotPoint::origin(), orientation: 0.0, bias: 0.0 }
    }

    pub fn transform(&self, point: &PlotPoint) -> PlotPoint {
        let shifted = point.sub(&self.position);
        let cos = self.orientation.cos();
        let sin = self.orientation.sin();
        let rotated = PlotPoint::new(
            shifted.x * cos - shifted.y * sin,
            shifted.x * sin + shifted.y * cos,
            shifted.z,
        );
        if self.bias == 0.0 {
            rotated
        } else {
            rotated.scale(1.0 + self.bias)
        }
    }

    pub fn inverse_transform(&self, point: &PlotPoint) -> PlotPoint {
        let unbiased = if self.bias == -1.0 {
            point.clone()
        } else {
            point.scale(1.0 / (1.0 + self.bias))
        };
        let cos = (-self.orientation).cos();
        let sin = (-self.orientation).sin();
        let unrotated = PlotPoint::new(
            unbiased.x * cos - unbiased.y * sin,
            unbiased.x * sin + unbiased.y * cos,
            unbiased.z,
        );
        unrotated.add(&self.position)
    }

    pub fn distance_to(&self, point: &PlotPoint) -> f64 {
        self.position.distance_to(point)
    }

    pub fn visibility(&self, point: &PlotPoint, max_range: f64) -> f64 {
        let d = self.distance_to(point);
        if d > max_range { 0.0 } else { 1.0 - d / max_range }
    }

    pub fn reliability(&self) -> f64 {
        1.0 - self.bias.abs()
    }

    pub fn combine(&self, other: &Narrator) -> Narrator {
        Narrator {
            position: self.position.midpoint(&other.position),
            orientation: (self.orientation + other.orientation) / 2.0,
            bias: (self.bias + other.bias) / 2.0,
        }
    }

    pub fn transform_arc(&self, arc: &Arc) -> Arc {
        Arc::new(
            self.transform(&arc.start),
            self.transform(&arc.end),
            arc.curvature * (1.0 + self.bias),
        )
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod plot_point_tests {
    use super::*;

    #[test]
    fn test_new() {
        let p = PlotPoint::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_origin() {
        let p = PlotPoint::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn test_distance_to_self() {
        let p = PlotPoint::new(3.0, 4.0, 0.0);
        assert!((p.distance_to(&p) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_345() {
        let a = PlotPoint::origin();
        let b = PlotPoint::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint() {
        let a = PlotPoint::new(0.0, 0.0, 0.0);
        let b = PlotPoint::new(4.0, 6.0, 8.0);
        let m = a.midpoint(&b);
        assert_eq!(m.x, 2.0);
        assert_eq!(m.y, 3.0);
        assert_eq!(m.z, 4.0);
    }

    #[test]
    fn test_magnitude() {
        let p = PlotPoint::new(3.0, 4.0, 0.0);
        assert!((p.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let p = PlotPoint::new(3.0, 4.0, 0.0);
        let n = p.normalize();
        assert!((n.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let p = PlotPoint::origin();
        let n = p.normalize();
        assert_eq!(n, PlotPoint::origin());
    }

    #[test]
    fn test_scale() {
        let p = PlotPoint::new(1.0, 2.0, 3.0);
        let s = p.scale(2.0);
        assert_eq!(s.x, 2.0);
        assert_eq!(s.y, 4.0);
        assert_eq!(s.z, 6.0);
    }

    #[test]
    fn test_add() {
        let a = PlotPoint::new(1.0, 2.0, 3.0);
        let b = PlotPoint::new(4.0, 5.0, 6.0);
        let c = a.add(&b);
        assert_eq!(c.x, 5.0);
        assert_eq!(c.y, 7.0);
        assert_eq!(c.z, 9.0);
    }

    #[test]
    fn test_sub() {
        let a = PlotPoint::new(4.0, 5.0, 6.0);
        let b = PlotPoint::new(1.0, 2.0, 3.0);
        let c = a.sub(&b);
        assert_eq!(c.x, 3.0);
        assert_eq!(c.y, 3.0);
        assert_eq!(c.z, 3.0);
    }

    #[test]
    fn test_dot() {
        let a = PlotPoint::new(1.0, 2.0, 3.0);
        let b = PlotPoint::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_cross() {
        let a = PlotPoint::new(1.0, 0.0, 0.0);
        let b = PlotPoint::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert_eq!(c.z, 1.0);
    }
}

#[cfg(test)]
mod arc_tests {
    use super::*;

    #[test]
    fn test_linear_arc_length() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        assert!((a.length() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_start() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        let p = a.interpolate(0.0);
        assert!((p.x - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_end() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        let p = a.interpolate(1.0);
        assert!((p.x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_mid() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        let p = a.interpolate(0.5);
        assert!((p.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse() {
        let a = Arc::linear(PlotPoint::new(1.0, 0.0, 0.0), PlotPoint::new(5.0, 0.0, 0.0));
        let r = a.reverse();
        assert_eq!(r.start, PlotPoint::new(5.0, 0.0, 0.0));
        assert_eq!(r.end, PlotPoint::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_degenerate() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::origin());
        assert!(a.is_degenerate());
    }

    #[test]
    fn test_not_degenerate() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(1.0, 0.0, 0.0));
        assert!(!a.is_degenerate());
    }

    #[test]
    fn test_midpoint() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        let m = a.midpoint();
        assert!((m.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_at() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        let (first, second) = a.split_at(0.5);
        assert!((first.end.x - 5.0).abs() < 1e-10);
        assert!((second.start.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_curved_arc_longer() {
        let straight = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        let curved = Arc::new(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0), 1.0);
        assert!(curved.length() >= straight.length());
    }

    #[test]
    fn test_interpolate_clamped() {
        let a = Arc::linear(PlotPoint::origin(), PlotPoint::new(10.0, 0.0, 0.0));
        let below = a.interpolate(-1.0);
        let above = a.interpolate(2.0);
        assert!((below.x).abs() < 1e-10);
        assert!((above.x - 10.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod tension_tests {
    use super::*;

    #[test]
    fn test_new() {
        let t = Tension::new(5.0, PlotPoint::new(1.0, 0.0, 0.0), 0.1);
        assert!((t.magnitude - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero() {
        let t = Tension::zero();
        assert!(t.is_resolved(0.01));
    }

    #[test]
    fn test_potential_energy() {
        let t = Tension::new(4.0, PlotPoint::new(1.0, 0.0, 0.0), 0.0);
        assert!((t.potential_energy() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay() {
        let t = Tension::new(10.0, PlotPoint::new(1.0, 0.0, 0.0), 1.0);
        let d = t.decay(1.0);
        assert!(d.magnitude < t.magnitude);
    }

    #[test]
    fn test_is_resolved() {
        let t = Tension::new(0.001, PlotPoint::new(1.0, 0.0, 0.0), 0.0);
        assert!(t.is_resolved(0.01));
    }

    #[test]
    fn test_not_resolved() {
        let t = Tension::new(5.0, PlotPoint::new(1.0, 0.0, 0.0), 0.0);
        assert!(!t.is_resolved(0.01));
    }

    #[test]
    fn test_add_tensions() {
        let a = Tension::new(3.0, PlotPoint::new(1.0, 0.0, 0.0), 0.1);
        let b = Tension::new(4.0, PlotPoint::new(1.0, 0.0, 0.0), 0.2);
        let c = a.add(&b);
        assert!(c.magnitude > a.magnitude);
    }

    #[test]
    fn test_escalate() {
        let t = Tension::new(5.0, PlotPoint::new(1.0, 0.0, 0.0), 0.0);
        let e = t.escalate(2.0);
        assert!((e.magnitude - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_redirect() {
        let t = Tension::new(5.0, PlotPoint::new(1.0, 0.0, 0.0), 0.0);
        let r = t.redirect(PlotPoint::new(0.0, 1.0, 0.0));
        assert!((r.direction.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_force_at_origin() {
        let t = Tension::new(5.0, PlotPoint::new(1.0, 0.0, 0.0), 0.0);
        let f = t.force_at(&PlotPoint::origin());
        assert!(f.magnitude() > 0.0);
    }

    #[test]
    fn test_force_decays_with_distance() {
        let t = Tension::new(5.0, PlotPoint::new(1.0, 0.0, 0.0), 0.0);
        let near = t.force_at(&PlotPoint::new(1.0, 0.0, 0.0));
        let far = t.force_at(&PlotPoint::new(100.0, 0.0, 0.0));
        assert!(near.magnitude() > far.magnitude());
    }

    #[test]
    fn test_direction_normalized() {
        let t = Tension::new(5.0, PlotPoint::new(3.0, 4.0, 0.0), 0.0);
        assert!((t.direction.magnitude() - 1.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod resolution_tests {
    use super::*;

    #[test]
    fn test_in_basin() {
        let r = Resolution::new(PlotPoint::origin(), 5.0, 0.5);
        assert!(r.is_in_basin(&PlotPoint::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn test_outside_basin() {
        let r = Resolution::new(PlotPoint::origin(), 5.0, 0.5);
        assert!(!r.is_in_basin(&PlotPoint::new(10.0, 0.0, 0.0)));
    }

    #[test]
    fn test_converge_step() {
        let r = Resolution::new(PlotPoint::new(10.0, 0.0, 0.0), 5.0, 0.5);
        let p = PlotPoint::origin();
        let next = r.converge_step(&p);
        assert!(next.x > p.x);
    }

    #[test]
    fn test_resolve_tension() {
        let r = Resolution::new(PlotPoint::origin(), 5.0, 0.5);
        let t = Tension::new(10.0, PlotPoint::new(1.0, 0.0, 0.0), 0.5);
        let resolved = r.resolve_tension(&t, 10);
        assert!(resolved.magnitude < t.magnitude);
    }

    #[test]
    fn test_time_to_resolve() {
        let r = Resolution::new(PlotPoint::origin(), 5.0, 0.5);
        let t = Tension::new(10.0, PlotPoint::new(1.0, 0.0, 0.0), 0.5);
        let time = r.time_to_resolve(&t, 0.01);
        assert!(time > 0.0);
    }

    #[test]
    fn test_stability() {
        let r = Resolution::new(PlotPoint::origin(), 5.0, 0.5);
        assert!((r.stability() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_merge() {
        let a = Resolution::new(PlotPoint::new(0.0, 0.0, 0.0), 5.0, 0.5);
        let b = Resolution::new(PlotPoint::new(10.0, 0.0, 0.0), 3.0, 0.3);
        let m = a.merge(&b);
        assert!((m.fixed_point.x - 5.0).abs() < 1e-10);
        assert!((m.basin_radius - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory() {
        let r = Resolution::new(PlotPoint::new(10.0, 0.0, 0.0), 5.0, 0.5);
        let traj = r.trajectory(&PlotPoint::origin(), 5);
        assert_eq!(traj.len(), 6);
    }

    #[test]
    fn test_trajectory_converges() {
        let r = Resolution::new(PlotPoint::new(10.0, 0.0, 0.0), 5.0, 0.5);
        let traj = r.trajectory(&PlotPoint::origin(), 20);
        let last = traj.last().unwrap();
        assert!(last.distance_to(&r.fixed_point) < PlotPoint::origin().distance_to(&r.fixed_point));
    }

    #[test]
    fn test_already_resolved_time() {
        let r = Resolution::new(PlotPoint::origin(), 5.0, 0.5);
        let t = Tension::new(0.001, PlotPoint::new(1.0, 0.0, 0.0), 0.5);
        assert_eq!(r.time_to_resolve(&t, 0.01), 0.0);
    }
}

#[cfg(test)]
mod narrator_tests {
    use super::*;

    #[test]
    fn test_omniscient() {
        let n = Narrator::omniscient();
        let p = PlotPoint::new(1.0, 2.0, 3.0);
        assert_eq!(n.transform(&p), p);
    }

    #[test]
    fn test_identity_transform() {
        let n = Narrator::new(PlotPoint::origin(), 0.0, 0.0);
        let p = PlotPoint::new(5.0, 5.0, 5.0);
        let t = n.transform(&p);
        assert!((t.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation() {
        let n = Narrator::new(PlotPoint::new(1.0, 0.0, 0.0), 0.0, 0.0);
        let p = PlotPoint::new(5.0, 0.0, 0.0);
        let t = n.transform(&p);
        assert!((t.x - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_90() {
        let n = Narrator::new(PlotPoint::origin(), PI / 2.0, 0.0);
        let p = PlotPoint::new(1.0, 0.0, 0.0);
        let t = n.transform(&p);
        assert!((t.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bias_scaling() {
        let n = Narrator::new(PlotPoint::origin(), 0.0, 1.0);
        let p = PlotPoint::new(1.0, 0.0, 0.0);
        let t = n.transform(&p);
        assert!((t.x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_roundtrip() {
        let n = Narrator::new(PlotPoint::new(3.0, 4.0, 0.0), 0.5, 0.2);
        let p = PlotPoint::new(10.0, 20.0, 30.0);
        let transformed = n.transform(&p);
        let recovered = n.inverse_transform(&transformed);
        assert!((recovered.x - p.x).abs() < 1e-10);
        assert!((recovered.y - p.y).abs() < 1e-10);
    }

    #[test]
    fn test_visibility_in_range() {
        let n = Narrator::new(PlotPoint::origin(), 0.0, 0.0);
        let vis = n.visibility(&PlotPoint::new(5.0, 0.0, 0.0), 10.0);
        assert!((vis - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_visibility_out_of_range() {
        let n = Narrator::new(PlotPoint::origin(), 0.0, 0.0);
        let vis = n.visibility(&PlotPoint::new(15.0, 0.0, 0.0), 10.0);
        assert_eq!(vis, 0.0);
    }

    #[test]
    fn test_reliability() {
        let n = Narrator::new(PlotPoint::origin(), 0.0, 0.3);
        assert!((n.reliability() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_combine() {
        let a = Narrator::new(PlotPoint::new(0.0, 0.0, 0.0), 0.0, 0.0);
        let b = Narrator::new(PlotPoint::new(10.0, 0.0, 0.0), PI, 0.5);
        let c = a.combine(&b);
        assert!((c.position.x - 5.0).abs() < 1e-10);
        assert!((c.orientation - PI / 2.0).abs() < 1e-10);
        assert!((c.bias - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_transform_arc() {
        let n = Narrator::new(PlotPoint::origin(), 0.0, 1.0);
        let arc = Arc::linear(PlotPoint::new(1.0, 0.0, 0.0), PlotPoint::new(2.0, 0.0, 0.0));
        let ta = n.transform_arc(&arc);
        assert!((ta.start.x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to() {
        let n = Narrator::new(PlotPoint::new(3.0, 4.0, 0.0), 0.0, 0.0);
        assert!((n.distance_to(&PlotPoint::origin()) - 5.0).abs() < 1e-10);
    }
}
