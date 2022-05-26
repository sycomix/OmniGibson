import os
from igibson import assets_path, app
from igibson.prims.prim_base import BasePrim
from igibson.systems.particle_system_base import BaseParticleSystem
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import classproperty, Serializable, assert_valid_key
from igibson.utils.sampling_utils import sample_cuboid_on_object
from igibson.utils.usd_utils import create_joint, array_to_vtarray
from igibson.utils.physx_utils import create_physx_particle_system, create_physx_particleset_pointinstancer, \
    bind_material
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.physx.scripts import particleUtils
from collections import OrderedDict
import numpy as np
from pxr import Gf, Vt, UsdShade, UsdGeom, PhysxSchema
import logging


class PhysxParticleInstancer(BasePrim):
    """
    Simple class that wraps the raw omniverse point instancer prim and provides convenience functions for
    particle access
    """
    def __init__(self, prim_path, name, idn):
        """
        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            idn (int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation.
        """
        # Store inputs
        self._idn = idn

        # Values loaded at runtime
        self._n_particles = None

        # Run super method directly
        super().__init__(prim_path=prim_path, name=name)

    def _load(self, simulator=None):
        # We raise an error, this should NOT be created from scratch
        raise NotImplementedError("PhysxPointInstancer should NOT be loaded via this class! Should be created before.")

    def _post_load(self, simulator=None):
        # Run super
        super()._post_load(simulator=simulator)

        # Store how many particles we have
        self._n_particles = len(self.particle_positions)

        # Set the invisibleIds to be 0, and all particles to be 1
        self.set_attribute(attr="ids", val=np.ones(self._n_particles))
        self.set_attribute(attr="invisibleIds", val=[0])

    def _initialize(self):
        # Run super first
        super()._initialize()

    def update_default_state(self):
        pass

    @property
    def n_particles(self):
        """
        Returns:
            int: Number of particles owned by this instancer
        """
        return self._n_particles

    @property
    def idn(self):
        """
        Returns:
            int: Identification number of this particle instancer
        """
        return self._idn

    @property
    def particle_group(self):
        """
        Returns:
            int: Particle group this instancer belongs to
        """
        return self.get_attribute(attr="physxParticle:particleGroup")

    @property
    def position(self):
        """
        Returns:
            3-array: (x,y,z) translation of this point instancer relative to its parent prim
        """
        return np.array(self.get_attribute(attr="xformOp:translate"))

    @position.setter
    def position(self, pos):
        """
        Sets this point instancer's (x,y,z) cartesian translation relative to its parent prim

        Args:
            pos (3-array): (x,y,z) relative position to set this prim relative to its parent prim
        """
        self.set_attribute(attr="xformOp:translate", val=Gf.Vec3f(*(pos.astype(float))))

    @property
    def particle_positions(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return np.array(self.get_attribute(attr="positions")) + self.position

    @particle_positions.setter
    def particle_positions(self, pos):
        """
        Set the particle positions for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired positions are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        pos = (pos - self.position).astype(float)
        self.set_attribute(attr="positions", val=array_to_vtarray(arr=pos, element_type=Gf.Vec3f))

    @property
    def particle_orientations(self):
        """
        Returns:
            np.array: (N, 4) numpy array, where each of the N particles' orientations are expressed in (x,y,z,w)
                quaternion coordinates relative to this instancer's parent prim
        """
        oris = self.get_attribute(attr="orientations")
        if oris is None:
            # Default orientations for all particles
            oris = np.zeros((self.n_particles, 4))
            oris[:, -1] = 1.0
        return np.array(oris)

    @particle_orientations.setter
    def particle_orientations(self, quat):
        """
        Set the particle positions for this instancer

        Args:
            np.array: (N, 4) numpy array, where each of the N particles' desired orientations are expressed in (x,y,z,w)
                quaternion coordinates relative to this instancer's parent prim
        """
        # Swap w position, since Quath takes (w,x,y,z)
        quat = quat[:, [3, 0, 1, 2]]
        self.set_attribute(attr="orientations", val=array_to_vtarray(arr=quat, element_type=Gf.Quath))

    @property
    def particle_velocities(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' velocities are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return np.array(self.get_attribute(attr="velocities"))

    @particle_velocities.setter
    def particle_velocities(self, vel):
        """
        Set the particle velocities for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired velocities are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        self.set_attribute(attr="velocities", val=array_to_vtarray(arr=vel, element_type=Gf.Vec3f))

    @property
    def particle_scales(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' scales are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        scales = self.get_attribute(attr="scales")
        return np.ones((self._n_particles, 3)) if scales is None else np.array(scales)

    @particle_scales.setter
    def particle_scales(self, scales):
        """
        Set the particle scales for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired scales are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        self.set_attribute(attr="scales", val=array_to_vtarray(arr=scales, element_type=Gf.Vec3f))

    @property
    def particle_prototype_ids(self):
        """
        Returns:
            np.array: (N,) numpy array, where each of the N particles' prototype_id (i.e.: which prototype is being used
                for that particle)
        """
        ids = self.get_attribute(attr="protoIndices")
        return np.zeros(self.n_particles) if ids is None else np.array(ids)

    @particle_prototype_ids.setter
    def particle_prototype_ids(self, prototype_ids):
        """
        Set the particle prototype_ids for this instancer

        Args:
            np.array: (N,) numpy array, where each of the N particles' desired prototype_id
                (i.e.: which prototype is being used for that particle)
        """
        self.set_attribute(attr="protoIndices", val=prototype_ids)

    @property
    def particle_visibilities(self):
        """
        Returns:
            np.array: (N,) numpy array, where each entry is the specific particle's visiblity
                (1 if visible, 0 otherwise)
        """
        # We leverage the ids + invisibleIds prim fields to infer visibility
        # id = 1 means visible, id = 0 means invisible
        ids = self.get_attribute("ids")
        return np.ones(self.n_particles) if ids is None else np.array(ids)

    @particle_visibilities.setter
    def particle_visibilities(self, visibilities):
        """
        Set the particle visibilities for this instancer

        Args:
            np.array: (N,) numpy array, where each entry is the specific particle's desired visiblity
                (1 if visible, 0 otherwise)
        """
        # We leverage the ids + invisibleIds prim fields to infer visibility
        # id = 1 means visible, id = 0 means invisible
        self.set_attribute(attr="ids", val=visibilities)

    def _dump_state(self):
        return OrderedDict(
            idn=self._idn,
            particle_group=self.particle_group,
            n_particles=self._n_particles,
            position=self.position,
            particle_positions=self.particle_positions,
            particle_velocities=self.particle_velocities,
            particle_orientations=self.particle_orientations,
            particle_scales=self.particle_scales,
            particle_prototype_ids=self.particle_prototype_ids,
        )

    def _load_state(self, state):
        # Sanity check the identification number and particle group
        assert self._idn == state["idn"], f"Got mismatch in identification number for this particle instancer when " \
            f"loading state! Should be: {self._idn}, got: {state['idn']}."
        assert self.particle_group == state["particle_group"], f"Got mismatch in particle group for this particle " \
            f"instancer when loading state! Should be: {self.particle_group}, got: {state['particle_group']}."

        # Set values appropriately
        self._n_particles = state["n_particles"]
        self.particle_positions = state["particle_positions"]
        self.particle_velocities = state["particle_velocities"]
        self.particle_orientations = state["particle_orientations"]
        self.particle_scales = state["particle_scales"]
        self.particle_prototype_ids = state["particle_prototype_ids"]

    def _serialize(self, state):
        # Compress into a 1D array
         return np.concatenate([
             [state["idn"], state["particle_group"], state["n_particles"]],
             state["position"],
             state["particle_positions"].reshape(-1),
             state["particle_velocities"].reshape(-1),
             state["particle_orientations"].reshape(-1),
             state["particle_scales"].reshape(-1),
             state["particle_prototype_ids"],
         ])

    def _deserialize(self, state):
        # Sanity check the identification number
        assert self._idn == state[0], f"Got mismatch in identification number for this particle instancer when " \
            f"deserializing state! Should be: {self._idn}, got: {state[0]}."
        assert self.particle_group == state[1], f"Got mismatch in particle group for this particle " \
            f"instancer when deserializing state! Should be: {self.particle_group}, got: {state[1]}."

        # De-compress from 1D array
        n_particles = int(state[2])
        state_dict = OrderedDict(
            idn=int(state[0]),
            particle_group=int(state[1]),
            n_particles=n_particles,
        )

        # Process remaining keys and reshape automatically
        keys = ("position", "particle_positions", "particle_velocities", "particle_orientations", "particle_scales", "particle_prototype_ids")
        sizes = ((3,), (n_particles, 3), (n_particles, 3), (n_particles, 4), (n_particles, 3), (n_particles,))

        idx = 3
        print(len(state))
        for key, size in zip(keys, sizes):
            length = np.product(size)
            state_dict[key] = state[idx: idx + length].reshape(size)
            idx += length

        return state_dict, idx


class MicroParticleSystem(BaseParticleSystem):
    """
    Global system for modeling "micro" level particles, e.g.: water, seeds, rice, etc. This system leverages
    Omniverse's native physx particle systems
    """
    # Particle system prim in the scene, should be generated at runtime
    prim = None

    # Particle prototypes -- will be list of mesh prims to use as particle prototypes for this system
    particle_prototypes = None

    # Particle instancers -- maps name to particle instancer prims (OrderedDict)
    particle_instancers = None

    # Particle material -- either a UsdShade.Material or None if no material is used for this particle system
    particle_material = None

    # Scaling factor to sample from when generating a new particle
    min_scale = None                # (x,y,z) scaling
    max_scale = None                # (x,y,z) scaling

    @classproperty
    def n_particles(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        return sum([instancer.n_particles for instancer in cls.particle_instancers.values()])

    @classproperty
    def prim_path(cls):
        """
        Returns:
            str: Path to this system's prim in the scene stage
        """
        return cls.prim.GetPrimPath().pathString

    @classmethod
    def initialize(cls, simulator):
        # Run super first
        super().initialize(simulator=simulator)

        # Initialize class variables that are mutable so they don't get overridden by children classes
        cls.particle_instancers = OrderedDict()

        # Set the default scales according to the collision offset
        cls.min_scale = np.ones(3) * cls.particle_contact_offset
        cls.max_scale = np.ones(3) * cls.particle_contact_offset

        # Create the particle system
        cls.prim = cls._create_particle_system()

        # Create the particle material
        cls.particle_material = cls._create_particle_material()
        if cls.particle_material is not None:
            # Move this material and standardize its naming scheme
            path_from = cls.particle_material.GetPrim().GetPrimPath().pathString
            mat_path = f"{cls.prim_path}/{cls.name}_material"
            omni.kit.commands.execute("MovePrim", path_from=path_from, path_to=mat_path)
            # Get updated reference to this material
            cls.particle_material = UsdShade.Material(get_prim_at_path(mat_path))

            # Bind this material to our particle system
            particleUtils.add_pbd_particle_material(simulator.stage, mat_path)
            bind_material(prim_path=cls.prim_path, material_path=mat_path)

        # Create the particle prototypes
        cls.particle_prototypes = cls._create_particle_prototypes()

    @classproperty
    def state_size(cls):
        # We have the number of particle instancers (1), the instancer groups, particle groups, and,
        # number of particles in each instancer (3n),
        # and the corresponding states in each instancer (X)
        return 1 + 3 * len(cls.particle_instancers) + sum(inst.state_size for inst in cls.particle_instancers.values())

    @classproperty
    def is_fluid(cls):
        """
        Returns:
            bool: Whether this system is modeling fluid or not
        """
        raise NotImplementedError()

    @classproperty
    def is_dynamic(cls):
        """
        Returns:
            bool: Whether this particle system should be dynamic, i.e.: subject to collisions and physics. If False,
                then generated particles will not move or collide
        """
        raise NotImplementedError()

    @classproperty
    def particle_contact_offset(cls):
        """
        Returns:
            float: Contact offset value to use for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#particle-system-configuration
                for more information
        """
        raise NotImplementedError()

    @classproperty
    def use_smoothing(cls):
        """
        Returns:
            bool: Whether to use smoothing or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#smoothing
                for more information
        """
        raise NotImplementedError()

    @classproperty
    def use_anisotropy(cls):
        """
        Returns:
            bool: Whether to use anisotropy or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#anisotropy
                for more information
        """
        raise NotImplementedError()

    @classproperty
    def use_isosurface(cls):
        """
        Returns:
            bool: Whether to use isosurface or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#isosurface
                for more information
        """
        raise NotImplementedError()

    @classproperty
    def particle_density(cls):
        """
        Returns:
            float: The per-particle density, in kg / m^3
        """
        raise NotImplementedError()

    @classmethod
    def _create_particle_prototypes(cls):
        """
        Creates any relevant particle prototypes to be used by this particle system.

        Returns:
            list of Usd.Prim: Mesh prim(s) to use as this system's particle prototype(s)
        """
        raise NotImplementedError()

    @classmethod
    def _create_particle_material(cls):
        """
        Creates the particle material to be used for this particle system. Prim path does not matter, as it will be
        overridden internally such that it is a child prim of this particle system's prim

        Returns:
            None or UsdShade.Material: If specified, is the material to apply to all particles. If None, no material
                will be used. Default is None
        """
        return None

    @classmethod
    def _create_particle_system(cls):
        """
        Creates the single, global particle system. This should only be ever called once, and during initialize()

        Returns:
            Usd.Prim: Particle system prim created
        """
        return create_physx_particle_system(
            prim_path=f"/World/{cls.name}",
            physics_scene_path=cls.simulator.get_physics_context().get_current_physics_scene_prim().GetPrimPath().pathString,
            particle_contact_offset=cls.particle_contact_offset,
            visual_only=cls.is_dynamic,
            smoothing=cls.use_smoothing,
            anisotropy=cls.use_anisotropy,
            isosurface=cls.use_isosurface,
        ).GetPrim()

    @classmethod
    def generate_particle_instancer(
            cls,
            n_particles,
            idn=None,
            particle_group=0,
            positions=None,
            velocities=None,
            orientations=None,
            scales=None,
            self_collision=True,
            prototype_indices=None,
    ):
        """
        Generates a new particle instancer with unique identification number @idn, and registers it internally

        Args:
            n_particles (int): Number of particles to generate for this instancer
            idn (None or int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation. If None, this system will generate a unique
                identifier automatically.
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by @self_collision
            positions (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions.
                If not specified, will be set to the origin by default
            velocities (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
                If not specified, all will be set to 0
            orientations (None or np.array): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (cls.min_scale, cls.max_scale)
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not
            prototype_indices (None or list of int): If specified, should specify which prototype should be used for
                each particle. If None, will use all 0s (i.e.: the first prototype created)

        Returns:
            PhysxParticleInstancer: Generated particle instancer
        """
        # Run sanity checks
        assert cls.initialized, "Must initialize system before generating particle instancers!"

        # Automatically generate an identification number for this instancer if none is specified
        if idn is None:
            max_idn = max([cls.particle_instancer_name_to_idn(name) for name in cls.particle_instancers.values()])
            idn = max_idn + 1

        # Generate standardized prim path for this instancer
        name = cls.particle_instancer_idn_to_name(idn=idn)

        # Create the instancer
        instance = create_physx_particleset_pointinstancer(
            name=name,
            particle_system_path=cls.prim_path,
            particle_group=particle_group,
            positions=np.zeros((n_particles, 3)) if positions is None else positions,
            self_collision=self_collision,
            fluid=cls.is_fluid,
            particle_mass=None,
            particle_density=cls.particle_density,
            orientations=orientations,
            velocities=velocities,
            angular_velocities=None,
            scales=np.random.uniform(cls.min_scale, cls.max_scale, size=(n_particles, 3)) if scales is None else scales,
            prototype_prim_paths=[pp.GetPrimPath().pathString for pp in cls.particle_prototypes],
            prototype_indices=prototype_indices,
            enabled=cls.is_dynamic,
        )

        # Create the instancer object that wraps the raw prim
        instancer = PhysxParticleInstancer(
            prim_path=instance.GetPrimPath().pathString,
            name=name,
            idn=idn,
        )
        instancer.initialize()
        cls.particle_instancers[name] = instancer

        return instancer

    @classmethod
    def generate_particle_instancer_from_mesh(
            cls,
            idn,
            particle_group,
            mesh_prim_path,
            sampling_distance=None,
            max_samples=5e5,
            sample_volume=True,
            self_collision=True,
            prototype_indices_choices=None,
    ):
        """
        Generates a new particle instancer with unique identification number @idn, with particles sampled from the mesh
        located at @mesh_prim_path, and registers it internally

        Args:
            idn (int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation.
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by @self_collision
            mesh_prim_path (str): Stage path to the mesh prim which will be converted into sampled particles
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            sample_volume (bool): Whether to sample the particles at the mesh's surface or throughout its entire volume
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not
            prototype_indices_choices (None or int or list of int): If specified, should specify which prototype(s)
                should be used for each particle. If None, will use all 0s (i.e.: the first prototype created). If a
                single number, will use that prototype ID for all sampled particles. If a list of int, will uniformly
                sample from those IDs for each particle.

        Returns:
            PhysxParticleInstancer: Generated particle instancer
        """
        # Run sanity checks
        assert cls.initialized, "Must initialize system before generating particle instancers!"

        # Generate standardized prim path for this instancer
        name = cls.particle_instancer_idn_to_name(idn=idn)
        instancer_prim_path = f"{cls.prim_path}/{name}"

        # Create points prim (this is used initially to generate the particles) and apply particle set API
        points_prim_path = f"{cls.prim_path}/tempSampledPoints"
        points = UsdGeom.Points.Define(cls.simulator.stage, points_prim_path).GetPrim()
        particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points)
        particle_set_api.CreateParticleSystemRel().SetTargets([cls.prim_path])

        # Apply the sampling API to our mesh prim and apply the sampling
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(get_prim_at_path(mesh_prim_path))
        sampling_api.CreateParticlesRel().AddTarget(points_prim_path)
        sampling_api.CreateSamplingDistanceAttr().Set(0 if sampling_distance is None else sampling_distance)
        sampling_api.CreateMaxSamplesAttr().Set(max_samples)
        sampling_api.CreateVolumeAttr().Set(sample_volume)

        # We apply 5 app steps to propagate the sampling (make sure to pause the sim since particles still propagate
        # forward even if we don't explicitly call sim.step())
        sim_is_stopped, sim_is_playing = cls.simulator.is_stopped(), cls.simulator.is_playing()
        cls.simulator.pause()
        for _ in range(5):
            app.update()
        if sim_is_stopped:
            cls.simulator.stop()
        elif sim_is_playing:
            cls.simulator.play()

        # Grab the actual positions, we will write this to a new instancer that's not tied to the sampler
        # The points / instancer tied to the sampled mesh seems to operate a bit weirdly, which is why we do it this way
        attr = "positions" if points.GetPrimTypeInfo().GetTypeName() == "PointInstancer" else "points"
        pos = points.GetAttribute(attr).Get()

        # Make sure sampling was successful
        assert len(pos) > 0, "Failed to sample particle points from mesh prim!"

        # Delete the points prim, we don't need it anymore
        cls.simulator.stage.RemovePrim(points_prim_path)

        # Get information about our sampled points
        n_particles = len(pos)
        if prototype_indices_choices is not None:
            prototype_indices = np.ones(n_particles, dtype=int) * prototype_indices_choices if \
                isinstance(prototype_indices_choices, int) else \
                np.random.choice(prototype_indices_choices, size=(n_particles,))
        else:
            prototype_indices = None

        # Create and return the generated instancer
        return cls.generate_particle_instancer(
            idn=idn,
            particle_group=particle_group,
            n_particles=n_particles,
            positions=pos,
            velocities=None,
            orientations=None,
            scales=None,
            self_collision=self_collision,
            prototype_indices=prototype_indices,
        )

    @classmethod
    def remove_particle_instancer(cls, name):
        """
        Removes particle instancer with name @name from this system.

        Args:
            name (str): Particle instancer name to remove. If it does not exist, then an error will be raised
        """
        # Make sure the instancer actually exists
        assert_valid_key(key=name, valid_keys=cls.particle_instancers, name="particle instancer")
        # Remove instancer from our tracking and delete its prim
        instancer = cls.particle_instancers.pop(name)
        cls.simulator.stage.RemovePrim(instancer.prim_path)

    @classmethod
    def particle_instancer_name_to_idn(cls, name):
        """
        Args:
            name (str): Particle instancer name

        Returns:
            int: Particle instancer identification number
        """
        return int(name.split(f"{cls.name}Instancer")[-1])

    @classmethod
    def particle_instancer_idn_to_name(cls, idn):
        """
        Args:
            idn (idn): Particle instancer identification number

        Returns:
            str: Name of the particle instancer auto-generated from its unique identification number
        """
        return f"{cls.name}Instancer{idn}"

    @classmethod
    def _sync_particle_instancers(cls, idns, particle_groups, particle_counts):
        """
        Synchronizes the particle instancers based on desired identification numbers @idns

        Args:
            idns (list of int): Desired unique instancers that should be active for this particle system
            particle_groups (list of int): Desired particle groups that each instancer should be. Length of this
                list should be the same length as @idns
            particle_counts (list of int): Desired particle counts that should exist per instancer. Length of this
                list should be the same length as @idns
        """
        # We have to be careful here -- some particle instancers may have been deleted / are mismatched, so we need
        # to update accordingly, potentially deleting stale instancers and creating new instancers as needed
        idn_to_info_mapping = {idn: {"group": group, "count": count}
                               for idn, group, count in zip(idns, particle_groups, particle_counts)}
        current_instancer_names = set(cls.particle_instancers.keys())
        desired_instancer_names = set(cls.particle_instancer_idn_to_name(idn=idn) for idn in idns)
        instancers_to_delete = current_instancer_names - desired_instancer_names
        instancers_to_create = desired_instancer_names - current_instancer_names
        common_instancers = current_instancer_names.intersection(desired_instancer_names)

        # Sanity check the common instancers, we will recreate any where there is a mismatch
        print(f"common: {common_instancers}")
        for name in common_instancers:
            idn = cls.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            instancer = cls.particle_instancers[name]
            if instancer.particle_group != info["group"] or instancer.n_particles != info["count"]:
                print(f"Got mismatch in particle instancer {name} when syncing, deleting and recreating instancer now.")
                # Add this instancer to both the delete and creation pile
                instancers_to_delete.add(name)
                instancers_to_create.add(name)

        # Delete any instancers we no longer want
        print(f"del: {instancers_to_delete}")
        for name in instancers_to_delete:
            instancer = cls.particle_instancers.pop(name)
            cls.simulator.stage.RemovePrim(instancer.prim_path)

        # Create any instancers we don't already have
        print(f"create: {instancers_to_create}")
        for name in instancers_to_create:
            idn = cls.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            cls.generate_particle_instancer(idn=idn, particle_group=info["group"], n_particles=info["count"])

    @classmethod
    def _dump_state(cls):
        return OrderedDict(
            n_particle_instancers=len(cls.particle_instancers),
            instancer_idns=[inst.idn for inst in cls.particle_instancers.values()],
            instancer_particle_groups=[inst.particle_group for inst in cls.particle_instancers.values()],
            instancer_particle_counts=[inst.n_particles for inst in cls.particle_instancers.values()],
            particle_states=OrderedDict(((name, inst.dump_state(serialized=False))
                                         for name, inst in cls.particle_instancers.items())),
        )

    @classmethod
    def _load_state(cls, state):
        # Synchronize the particle instancers
        cls._sync_particle_instancers(
            idns=state["instancer_idns"],
            particle_groups=state["instancer_particle_groups"],
            particle_counts=state["instancer_particle_counts"],
        )

        # Iterate over all particle states and load their respective states
        for name, inst_state in state["particle_states"].items():
            cls.particle_instancers[name].load_state(inst_state, serialized=False)

    @classmethod
    def _serialize(cls, state):
        # Array is number of particle instancers, then the corresponding states for each particle instancer
        return np.concatenate([
            [state["n_particle_instancers"]],
            state["instancer_idns"],
            state["instancer_particle_groups"],
            state["instancer_particle_counts"],
            *[cls.particle_instancers[name].serialize(inst_state)
              for name, inst_state in state["particle_states"].items()],
        ])

    @classmethod
    def _deserialize(cls, state):
        # Synchronize the particle instancers
        n_instancers = int(state[0])
        instancer_info = OrderedDict()
        idx = 1
        for info_name in ("instancer_idns", "instancer_particle_groups", "instancer_particle_counts"):
            instancer_info[info_name] = state[idx: idx + n_instancers].astype(int).tolist()
            idx += n_instancers
        print(f"Syncing particles. with {n_instancers} instancers..")
        cls._sync_particle_instancers(
            idns=instancer_info["instancer_idns"],
            particle_groups=instancer_info["instancer_particle_groups"],
            particle_counts=instancer_info["instancer_particle_counts"],
        )

        # Procedurally deserialize the particle states
        particle_states = OrderedDict()
        print(f"total state size: {len(state)}")
        for idn in instancer_info["instancer_idns"]:
            print(f"Deserializing {idn}...")
            name = cls.particle_instancer_idn_to_name(idn=idn)
            state_size = cls.particle_instancers[name].state_size
            print(f"state size: {state_size}")
            particle_states[name] = cls.particle_instancers[name].deserialize(state[idx: idx + state_size])
            idx += state_size

        return OrderedDict(
            n_particle_instancers=n_instancers,
            **instancer_info,
            particle_states=particle_states,
        ), idx

    @classmethod
    def set_scale_limits(cls, minimum=None, maximum=None):
        """
        Set the min and / or max scaling limits that will be uniformly sampled from when generating new particles

        Args:
            minimum (None or 3-array): If specified, should be (x,y,z) minimum scaling factor to apply to generated
                particles
            maximum (None or 3-array): If specified, should be (x,y,z) maximum scaling factor to apply to generated
                particles
        """
        if minimum is not None:
            cls.min_scale = np.array(minimum)
        if maximum is not None:
            cls.max_scale = np.array(maximum)

    @classmethod
    def remove_all_particle_instancers(cls):
        """
        Removes all particle instancers and deletes them from the simulator
        """
        cls._sync_particle_instancers(idns=[], particle_groups=[], particle_counts=[])


class FluidSystem(MicroParticleSystem):
    """
    Particle system class simulating fluids, leveraging isosurface feature in omniverse to render nice PBR fluid texture
    """

    @classproperty
    def is_fluid(cls):
        return True

    @classproperty
    def is_dynamic(cls):
        return True

    @classproperty
    def use_smoothing(cls):
        return False

    @classproperty
    def use_anisotropy(cls):
        return False

    @classproperty
    def use_isosurface(cls):
        # TODO: Make true once omni bugs are fixed
        return False


class WaterSystem(FluidSystem):
    """
    Particle system class to simulate water. Uses a transparent material and isosurface to render the water
    """
    @classproperty
    def _register_system(cls):
        # We should register this system since it's an "actual" system (not an intermediate class)
        return True

    @classproperty
    def particle_contact_offset(cls):
        return 0.004

    @classproperty
    def particle_density(cls):
        # Water is 1000 kg/m^3
        return 1000.0

    @classmethod
    def _create_particle_prototypes(cls):
        # Simulate water particles with simple spheres
        prototype = UsdGeom.Sphere.Define(cls.simulator.stage, f"{cls.prim_path}/waterParticlePrototype")
        prototype.CreateRadiusAttr().Set(0.99 * 0.6 * cls.particle_contact_offset)
        return [prototype.GetPrim()]

    @classmethod
    def _create_particle_material(cls):
        # Use DeepWater omni present for rendering water
        mtl_created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniSurfacePresets.mdl",
            mtl_name="OmniSurface_DeepWater",
            mtl_created_list=mtl_created,
        )
        material_path = mtl_created[0]

        # Also apply physics to this material
        particleUtils.add_pbd_particle_material(cls.simulator.stage, material_path)

        # Return generated material
        return UsdShade.Material(get_prim_at_path(material_path))