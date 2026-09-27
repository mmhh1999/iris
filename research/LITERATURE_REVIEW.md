# Literature review — calibrated sunlight as a constraint for indoor inverse rendering

2026-09-27. Narrative synthesis. The per-paper table with links, how far each paper was
read, and the feature columns is `NOVELTY_AUDIT.md`. Every work cited here was checked
this session against the link given there; nothing is cited from memory alone.

## 1. The problem family we would extend: multi-view indoor inverse rendering

IRIS (CVPR 2025) and its predecessor FIPT (ICCV 2023) recover materials and emitters from
posed multi-view images of a room with known geometry. They factor light transport into
mesh emitters plus a cached surface light field.

Neither has any light source outside the reconstructed mesh: a ray that leaves the room
returns zero (`IRIS_ARCHITECTURE_AUDIT.md` §3, §12). When the sun comes through a window,
the only things that can explain a sun patch are the albedo or an "emitter" painted on the
floor. We measured this directly: +19.5 / +14.3 percentage points of extra albedo in sunlit
floor regions (EXP0019, EXP0027).

The same limitation holds for:
- TexIR (CVPR 2023), whose windows are HDR textures on the mesh;
- MAIR (CVPR 2023), which uses a 3D lighting volume;
- the 3DGS-era indoor methods: AEGIR (2026), SGS-Intrinsic (CVPR 2026), IR-HGP (CVPR 2026).

The last two go furthest in suppressing baked shadows, with a learned deshadowing model or
a generative illumination prior. On our uniform-floor benchmark SGS-Intrinsic already
reduces sun contamination to about 1 point (EXP0024). **"Sun patches contaminate albedo"
is therefore a known and largely addressed failure, not an open problem.**

## 2. Explicit sun indoors

Three lines of work already put an explicit sun indoors:
- **Li et al., ECCV 2022** model a window's radiance as three spherical Gaussians (sun, sky,
  ground), from a single image. When fitting window light for their data, the sun direction
  is taken as the brightest direction of a panorama rendered facing out of the window, and
  then fixed. Nothing is calibrated from time or place.
- **ProjectiveShading (CGF 2026)** estimates the sun direction and a "sunlight map" from one
  indoor image, for object insertion.
- **Krawez et al., RA-L 2021** are the closest work we found. They:
  - build an indoor model lit by a sun/sky/ground outdoor model;
  - compute the sun position from the scene's geolocation, the current time and a *given*
    scene orientation;
  - precompute window transport by radiosity and rescale the components per frame;
  - evaluate on real RGB-D sequences over many times of day, seasons and weather.

  Their reflectance map comes from a separate capture under lamps after sunset (Krawez et al.,
  IROS 2018). The sun is used to *predict* appearance for camera tracking, never to *recover*
  materials.

Building-science daylight simulation, with daylight coefficients (Tregenza & Waters 1983),
does the same forward computation.

**So "calibrated ephemeris sun through a window, predicting indoor appearance at other times"
is established. What remains open is the inverse direction.**

Ji, Sawyer and Narasimhan (ISVC 2023; MVA 2024; ISVC 2025) capture indoor HDR panoramas
together with outdoor hemispherical HDR photos (the Cali-HDR dataset we use). They relight
one panorama under edited sun positions. That is also a single capture and forward relighting.

## 3. Calibrated sun over time as supervision, outdoors

The inverse use of a known, moving sun is well established outdoors:
- **Satellite NeRFs.** S-NeRF (CVPRW 2021), Sat-NeRF (CVPRW 2022) and EO-NeRF (CVPRW 2023)
  take the per-image sun angles as given. They use multi-date imagery to separate cast
  shadows from albedo and to sharpen geometry. SUNDIAL (CVPRW 2024) inverts this and recovers
  the sun and the capture time from shadows.
- **Webcam time-lapse.**
  - Heliometric stereo (ECCV 2012) computes the sun direction from GPS and timestamps and
    runs photometric stereo on time-lapse images.
  - Factored time-lapse video (SIGGRAPH 2007) splits a day's sequence into sun, sky, shadow
    and reflectance.
  - Hold-Geoffroy et al. (ICCP 2015) show a limitation. On clear days the sun's path is close
    to a plane, so normals are poorly conditioned from photometric cues alone; partly cloudy
    days are better.
- **Outdoor radiance fields.** SOL-NeRF, SR-TensoRF, NeRF-OSR, UrbanIR, GaRe, OSDR-GS and
  ROS-GS split sun from sky, mostly with an image-derived sun. NeRF-OSR supplies
  multi-session real ground truth.

## 4. The general principle: more illumination conditions, less ambiguity

Eclipse (CVPR 2024) exploits unintended shadows, and Dynamic Inverse Rendering (ECCV 2026)
exploits object motion. Both state or rely on the general fact that diverse lighting
constrains materials. GLOW (2025) applies it indoors with a calibrated, moving co-located
flash. **"Multiple lighting conditions help" cannot itself be a contribution.**

## 5. What is left, and how to test it

Not found, though a failed search does not prove absence: **multi-view indoor inverse
rendering supervised by the calibrated sun's motion through the window over several times of
day, validated on held-out-time photographs.** This transfers §3 into §1.

Its claimable content would be:
1. window-aperture transport with interreflection, which satellites do not need;
2. calibrating the one yaw DOF from patches, where Krawez assumes it is given;
3. showing that *calibration* beats an explicit sun estimated from images (Li 2022 /
   ProjectiveShading style).

Point (3) is the decisive comparison. It is what T11 (synthetic, EXP0030–0036) and EXP0037
(real patch prediction) test.

Our own results so far bear on this:
- In a controlled room, the ambiguity that survives an explicit sun lives in the **sky/ambient
  term**, not in the sun (D0020).
- Material recovery is extremely sensitive to sun direction. A 1.6° error is enough to bake
  patches back into albedo (EXP0035).
- So calibration is worth something only if it is accurate to a fraction of a degree, or when
  image estimation is hard.
