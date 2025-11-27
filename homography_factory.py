"""
Factory for creating homography providers based on configuration.

This module provides a factory pattern for instantiating homography providers
from configuration. It supports:
- Direct creation by approach type
- Creation from configuration objects
- Provider registration for extensibility
- Fallback chain handling for robustness
"""

from typing import Optional, Type, Dict, Any
import logging

from homography_interface import HomographyProvider, HomographyApproach
from homography_config import HomographyConfig
from intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography
from feature_match_homography import FeatureMatchHomography
from learned_homography import LearnedHomography


# Configure logging
logger = logging.getLogger(__name__)


class HomographyFactory:
    """Factory for creating homography providers.

    Supports:
    - Creating providers by approach type
    - Loading from configuration
    - Fallback chain for robustness
    - Custom provider registration

    The factory maintains a registry mapping HomographyApproach enum values
    to provider classes. New providers can be registered at runtime.

    Example:
        >>> # Create provider directly
        >>> provider = HomographyFactory.create(
        ...     HomographyApproach.INTRINSIC_EXTRINSIC,
        ...     width=2560,
        ...     height=1440
        ... )
        >>>
        >>> # Create from configuration
        >>> config = HomographyConfig.from_yaml('config.yaml')
        >>> provider = HomographyFactory.from_config(config, width=2560, height=1440)
        >>>
        >>> # Register custom provider
        >>> HomographyFactory.register(
        ...     HomographyApproach.LEARNED,
        ...     MyCustomLearnedProvider
        ... )

    Class Attributes:
        _registry: Dictionary mapping HomographyApproach to provider classes
    """

    _registry: Dict[HomographyApproach, Type[HomographyProvider]] = {
        HomographyApproach.INTRINSIC_EXTRINSIC: IntrinsicExtrinsicHomography,
        HomographyApproach.FEATURE_MATCH: FeatureMatchHomography,
        HomographyApproach.LEARNED: LearnedHomography,
    }

    @classmethod
    def create(
        cls,
        approach: HomographyApproach,
        width: int,
        height: int,
        **kwargs
    ) -> HomographyProvider:
        """Create a homography provider for the specified approach.

        This method instantiates a provider class based on the approach type.
        Additional keyword arguments are passed to the provider's constructor.

        Args:
            approach: Homography approach to use (e.g., INTRINSIC_EXTRINSIC)
            width: Image width in pixels
            height: Image height in pixels
            **kwargs: Additional provider-specific parameters:
                For IntrinsicExtrinsicHomography:
                    - pixels_per_meter: Scale for map visualization (default: 100.0)
                For FeatureMatchHomography:
                    - detector: Feature detector ('sift', 'orb', 'loftr')
                    - min_matches: Minimum matches required (default: 10)
                    - ransac_threshold: RANSAC threshold in pixels (default: 3.0)
                For LearnedHomography:
                    - model_path: Path to trained model
                    - model_type: Model architecture ('homography_net', etc.)
                    - device: Compute device ('cpu', 'cuda', 'mps')

        Returns:
            HomographyProvider instance ready for use

        Raises:
            ValueError: If approach is not registered or parameters are invalid
            RuntimeError: If provider instantiation fails

        Example:
            >>> # Create intrinsic/extrinsic provider
            >>> provider = HomographyFactory.create(
            ...     HomographyApproach.INTRINSIC_EXTRINSIC,
            ...     width=2560,
            ...     height=1440,
            ...     pixels_per_meter=100.0
            ... )
            >>>
            >>> # Create feature matching provider
            >>> provider = HomographyFactory.create(
            ...     HomographyApproach.FEATURE_MATCH,
            ...     width=2560,
            ...     height=1440,
            ...     detector='sift',
            ...     min_matches=10
            ... )
        """
        if approach not in cls._registry:
            registered = ', '.join(a.value for a in cls._registry.keys())
            raise ValueError(
                f"Approach '{approach.value}' not registered. "
                f"Available approaches: {registered}"
            )

        provider_class = cls._registry[approach]

        try:
            provider = provider_class(width=width, height=height, **kwargs)
            logger.info(
                f"Created {approach.value} provider "
                f"(class: {provider_class.__name__}, "
                f"resolution: {width}x{height})"
            )
            return provider
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for {approach.value} provider: {e}\n"
                f"Provider class: {provider_class.__name__}\n"
                f"Parameters: width={width}, height={height}, kwargs={kwargs}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {approach.value} provider: {e}"
            ) from e

    @classmethod
    def from_config(
        cls,
        config: HomographyConfig,
        width: int,
        height: int,
        try_fallbacks: bool = True
    ) -> HomographyProvider:
        """Create a provider from configuration.

        This method creates a provider based on the configuration's primary
        approach. If try_fallbacks is True and the primary approach fails,
        it will attempt to create providers using the fallback chain.

        Args:
            config: HomographyConfig object specifying approach and parameters
            width: Image width in pixels
            height: Image height in pixels
            try_fallbacks: If True, attempt fallback approaches on failure
                (default: True)

        Returns:
            HomographyProvider instance created from configuration

        Raises:
            ValueError: If no provider could be created (including fallbacks)
            RuntimeError: If provider creation fails

        Example:
            >>> config = HomographyConfig.from_yaml('config.yaml')
            >>> provider = HomographyFactory.from_config(
            ...     config,
            ...     width=2560,
            ...     height=1440
            ... )
            >>>
            >>> # Without fallbacks (fail fast)
            >>> provider = HomographyFactory.from_config(
            ...     config,
            ...     width=2560,
            ...     height=1440,
            ...     try_fallbacks=False
            ... )
        """
        # Try primary approach
        primary_approach = config.approach
        primary_config = config.get_approach_config(primary_approach)

        logger.info(
            f"Attempting to create provider with primary approach: "
            f"{primary_approach.value}"
        )

        try:
            provider = cls.create(
                primary_approach,
                width=width,
                height=height,
                **primary_config
            )
            logger.info(f"Successfully created {primary_approach.value} provider")
            return provider

        except Exception as e:
            logger.warning(
                f"Failed to create {primary_approach.value} provider: {e}"
            )

            if not try_fallbacks or not config.fallback_approaches:
                raise RuntimeError(
                    f"Failed to create {primary_approach.value} provider and "
                    f"no fallbacks available: {e}"
                ) from e

            # Try fallback approaches in order
            for fallback_approach in config.fallback_approaches:
                logger.info(
                    f"Attempting fallback approach: {fallback_approach.value}"
                )

                fallback_config = config.get_approach_config(fallback_approach)

                try:
                    provider = cls.create(
                        fallback_approach,
                        width=width,
                        height=height,
                        **fallback_config
                    )
                    logger.info(
                        f"Successfully created fallback provider: "
                        f"{fallback_approach.value}"
                    )
                    return provider

                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback approach {fallback_approach.value} also failed: "
                        f"{fallback_error}"
                    )
                    continue

            # All approaches failed
            attempted = [primary_approach] + config.fallback_approaches
            attempted_str = ', '.join(a.value for a in attempted)
            raise ValueError(
                f"Failed to create provider with any configured approach. "
                f"Attempted: {attempted_str}. "
                f"Check configuration and ensure required dependencies are available."
            ) from e

    @classmethod
    def register(
        cls,
        approach: HomographyApproach,
        provider_class: Type[HomographyProvider]
    ) -> None:
        """Register a new provider class for an approach.

        This allows extending the factory with custom provider implementations.
        The registered class must implement the HomographyProvider interface
        and accept at minimum (width, height) constructor parameters.

        Args:
            approach: HomographyApproach enum value to associate with this provider
            provider_class: Class implementing HomographyProvider interface

        Raises:
            ValueError: If provider_class is not a subclass of HomographyProvider

        Example:
            >>> class MyCustomProvider(HomographyProvider):
            ...     def __init__(self, width: int, height: int, **kwargs):
            ...         # Custom implementation
            ...         pass
            ...
            >>> HomographyFactory.register(
            ...     HomographyApproach.LEARNED,
            ...     MyCustomProvider
            ... )
            >>> provider = HomographyFactory.create(
            ...     HomographyApproach.LEARNED,
            ...     width=2560,
            ...     height=1440
            ... )
        """
        if not isinstance(provider_class, type):
            raise ValueError(
                f"provider_class must be a class, got {type(provider_class)}"
            )

        if not issubclass(provider_class, HomographyProvider):
            raise ValueError(
                f"provider_class must be a subclass of HomographyProvider, "
                f"got {provider_class.__name__}"
            )

        # Check if overwriting existing registration
        if approach in cls._registry:
            old_class = cls._registry[approach]
            logger.warning(
                f"Overwriting existing registration for {approach.value}: "
                f"{old_class.__name__} -> {provider_class.__name__}"
            )

        cls._registry[approach] = provider_class
        logger.info(
            f"Registered {provider_class.__name__} for approach {approach.value}"
        )

    @classmethod
    def get_registered_approaches(cls) -> list[HomographyApproach]:
        """Get list of registered approaches.

        Returns:
            List of HomographyApproach enum values that have registered providers

        Example:
            >>> approaches = HomographyFactory.get_registered_approaches()
            >>> print([a.value for a in approaches])
            ['intrinsic_extrinsic', 'feature_match', 'learned']
        """
        return list(cls._registry.keys())

    @classmethod
    def get_provider_class(cls, approach: HomographyApproach) -> Type[HomographyProvider]:
        """Get the provider class registered for an approach.

        Args:
            approach: HomographyApproach to look up

        Returns:
            Provider class associated with this approach

        Raises:
            ValueError: If approach is not registered

        Example:
            >>> provider_class = HomographyFactory.get_provider_class(
            ...     HomographyApproach.INTRINSIC_EXTRINSIC
            ... )
            >>> print(provider_class.__name__)
            'IntrinsicExtrinsicHomography'
        """
        if approach not in cls._registry:
            registered = ', '.join(a.value for a in cls._registry.keys())
            raise ValueError(
                f"Approach '{approach.value}' not registered. "
                f"Available approaches: {registered}"
            )

        return cls._registry[approach]
