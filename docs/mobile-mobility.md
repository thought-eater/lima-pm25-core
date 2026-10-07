# Mobile-sensor mobility

## Role

The mobility contract documents and enforces the restrictions applied to
externally supplied mobile-sensor positions. It validates positions; it does not
select a route, action, reward, or navigation policy.

`MobilityPolicy` is immutable during a run. Changing an experimental setting
means creating another policy instance. This preserves the configuration used by
an existing run without preventing later experiments from selecting different
values.

## Road mobility policy

Construct `MobilityPolicy(road_tolerance_m=...)` with an explicitly selected
position-to-road tolerance in meters. This value has no scientific default; the
scenario must explain why its selected tolerance is appropriate. The supplied
`SpatialDomain` provides the projected OSM `drive` graph and its exterior buffer,
so the policy does not repeat those settings.

`maximum_speed_m_s` is optional and has no scientific default. Leaving it as
`None` performs no speed validation. A study scenario that selects a speed must
record its justification rather than treating a test value as a Lima constant.

Distances use meters, elapsed time uses seconds, and speed uses meters per
second, as indicated by the parameter names. Invalid trajectories are rejected;
positions are never snapped to a road or replaced.

## Study boundary and road buffer

The concentration grid remains tied to the configured study boundary. The road
network is downloaded over a slightly larger WGS84 bounding box so a vehicle can
follow connected streets just outside that grid. The initial configurable
`road_network_buffer_m` is 50 m.

The default southern study boundary is extended to `-12.079` so the fixed Campo
de Marte site is not disproportionately close to the southern edge. Fixed-site
coordinates remain the source-qualified coordinates of their real buildings;
changing a study boundary never changes a station coordinate.

The 50 m road-network buffer and the road-position tolerance are different:

- the domain-owned buffer controls how far the downloaded mobility network and
  permitted excursions extend beyond the concentration domain;
- the tolerance controls how far a supplied position may be from an OSM road
  geometry.

Both are configuration, not calibrated scientific constants.

## Supplied trajectories

A `TimestampedTrajectoryProvider` consumes immutable `TrajectoryPoint` records
for one sensor. Times must increase strictly and coordinates must be finite in
the projected domain CRS. Every sensor due time must have an explicit recorded
position. The provider does not interpolate, extrapolate, snap, or repair a
missing position.

Each supplied position is checked against the selected road tolerance and
the domain's `road_network_buffer_m`. A maximum speed is checked only when the scenario has
configured one. There is no maximum-gap rule: position availability is verified
directly at the sensor due times.

The provider has a `provider_id`, used as `position_source_id` in each processed
due-time `MeasurementRecord`. Its value identifies the provider, not the
content of an external trajectory file. An optional input `resource_id` and
content hash identify that file once in the run or environment bundle when
available. The provider remains stateless after validation;
each processed position is retained in chronological measurement history, so
successful and unsupported attempts do not require a duplicate mutable
`used_positions` history.
