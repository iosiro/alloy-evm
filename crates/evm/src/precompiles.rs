//! Helpers for dealing with Precompiles.
use alloc::{boxed::Box, string::String, sync::Arc, vec};
use core::fmt::Debug;
use revm::{
    context::{Cfg, ContextTr, LocalContextTr},
    handler::{EthPrecompiles, PrecompileProvider},
    interpreter::{CallInput, Gas, InputsImpl, InstructionResult, InterpreterResult},
    precompile::{PrecompileError, PrecompileFn, PrecompileId, PrecompileResult, Precompiles},
    primitives::{
        hardfork::SpecId,
        map::{HashMap, HashSet},
        Address, Bytes, U256,
    },
};

use crate::{Database, EvmInternals};

/// A mapping of precompile contracts that can be either static (builtin) or dynamic.
///
/// This is an optimization that allows us to keep using the static precompiles
/// until we need to modify them, at which point we convert to the dynamic representation.
#[derive(Clone)]
pub struct PrecompilesMap<CTX: ContextTr, P: PrecompileProvider<CTX>> {
    /// The wrapped precompiles in their current representation.
    inner: P,

    dyn_precompiles: DynPrecompiles,
    /// An optional dynamic precompile loader that can lookup precompiles dynamically.
    lookup: Option<Arc<dyn PrecompileLookup>>,

    _marker: core::marker::PhantomData<CTX>,
}

impl<CTX: ContextTr> PrecompilesMap<CTX, EthPrecompiles> {
    /// Creates a new set of precompiles from the static ethereum precompiles.
    pub fn from_static(static_precompiles: &'static Precompiles) -> Self {
        Self::new(EthPrecompiles { precompiles: static_precompiles, spec: SpecId::default() })
    }
}

impl<CTX: ContextTr, P: PrecompileProvider<CTX>> PrecompilesMap<CTX, P> {
    /// Creates a new set of precompiles for a spec.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            dyn_precompiles: DynPrecompiles::default(),
            lookup: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Applies a transformation to the precompile at the given address.
    ///
    /// This method allows you to add, update, or remove a precompile by applying a closure
    /// to the existing precompile (if any) at the specified address.
    ///
    /// # Behavior
    ///
    /// The closure receives:
    /// - `Some(precompile)` if a precompile exists at the address
    /// - `None` if no precompile exists at the address
    ///
    /// Based on what the closure returns:
    /// - `Some(precompile)` - Insert or replace the precompile at the address
    /// - `None` - Remove the precompile from the address (if it exists)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Add a new precompile
    /// precompiles.apply_precompile(&address, |_| Some(my_precompile));
    ///
    /// // Update an existing precompile
    /// precompiles.apply_precompile(&address, |existing| {
    ///     existing.map(|p| wrap_with_logging(p))
    /// });
    ///
    /// // Remove a precompile
    /// precompiles.apply_precompile(&address, |_| None);
    ///
    /// // Conditionally update
    /// precompiles.apply_precompile(&address, |existing| {
    ///     if let Some(p) = existing {
    ///         Some(modify_precompile(p))
    ///     } else {
    ///         Some(create_default_precompile())
    ///     }
    /// });
    /// ```
    pub fn apply_precompile<F>(&mut self, address: &Address, f: F)
    where
        F: FnOnce(Option<DynPrecompile>) -> Option<DynPrecompile>,
    {
        let current = self.dyn_precompiles.inner.get(address).cloned();

        // apply the transformation function
        let result = f(current);

        match result {
            Some(transformed) => {
                // insert the transformed precompile
                self.dyn_precompiles.inner.insert(*address, transformed);
                self.dyn_precompiles.addresses.insert(*address);
            }
            None => {
                // remove the precompile if the transformation returned None
                self.dyn_precompiles.inner.remove(address);
                self.dyn_precompiles.addresses.remove(address);
            }
        }
    }

    /// Builder-style method that applies a transformation to the precompile at the given address.
    ///
    /// This is a consuming version of [`apply_precompile`](Self::apply_precompile) that returns
    /// `Self`. See [`apply_precompile`](Self::apply_precompile) for detailed behavior and
    /// examples.
    pub fn with_applied_precompile<F>(mut self, address: &Address, f: F) -> Self
    where
        F: FnOnce(Option<DynPrecompile>) -> Option<DynPrecompile>,
    {
        self.apply_precompile(address, f);
        self
    }

    /// Sets a dynamic precompile lookup function that is called for addresses not found
    /// in the static precompile map.
    ///
    /// This method allows you to provide runtime-resolved precompiles that aren't known
    /// at initialization time. The lookup function is called whenever a precompile check
    /// is performed for an address that doesn't exist in the main precompile map.
    ///
    /// # Important Notes
    ///
    /// - **Priority**: Static precompiles take precedence. The lookup function is only called if
    ///   the address is not found in the main precompile map.
    /// - **Gas accounting**: Addresses resolved through this lookup are always treated as cold,
    ///   meaning they incur cold access costs even on repeated calls within the same transaction.
    ///   See also [`PrecompileProvider::warm_addresses`].
    /// - **Performance**: The lookup function is called on every precompile check for
    ///   non-registered addresses, so it should be efficient.
    ///
    /// # Example
    ///
    /// ```ignore
    /// precompiles.set_precompile_lookup(|address| {
    ///     // Dynamically resolve precompiles based on address pattern
    ///     if address.as_slice().starts_with(&[0xDE, 0xAD]) {
    ///         Some(DynPrecompile::new(|input| {
    ///             // Custom precompile logic
    ///             Ok(PrecompileOutput {
    ///                 gas_used: 100,
    ///                 bytes: Bytes::from("dynamic precompile"),
    ///             })
    ///         }))
    ///     } else {
    ///         None
    ///     }
    /// });
    /// ```
    pub fn set_precompile_lookup<L>(&mut self, lookup: L)
    where
        L: PrecompileLookup + 'static,
    {
        self.lookup = Some(Arc::new(lookup));
    }

    /// Builder-style method to set a dynamic precompile lookup function.
    ///
    /// This is a consuming version of [`set_precompile_lookup`](Self::set_precompile_lookup)
    /// that returns `Self` for method chaining.
    ///
    /// See [`set_precompile_lookup`](Self::set_precompile_lookup) for detailed behavior,
    /// important notes, and examples.
    pub fn with_precompile_lookup<L>(mut self, lookup: L) -> Self
    where
        L: PrecompileLookup + 'static,
    {
        self.set_precompile_lookup(lookup);
        self
    }

    /// Gets a reference to the precompile at the given address.
    ///
    /// This method first checks the static precompile map, and if not found,
    /// falls back to the dynamic lookup function (if set).
    pub fn contains(&self, address: &Address) -> bool {
        // First, check the static precompiles
        if self.inner.contains(address) || self.dyn_precompiles.addresses.contains(address) {
            return true;
        }

        if let Some(lookup) = self.lookup.as_ref() {
            return lookup.lookup(address).is_some();
        }

        false
    }
}

impl<CTX: ContextTr, P: PrecompileProvider<CTX> + Debug> From<P> for PrecompilesMap<CTX, P> {
    fn from(value: P) -> Self {
        Self::new(value)
    }
}

impl<CTX: ContextTr, P: PrecompileProvider<CTX> + Debug> Debug for PrecompilesMap<CTX, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PrecompilesMap")
            .field("inner", &self.inner)
            .field("dyn_precompiles", &self.dyn_precompiles)
            .finish()
    }
}
impl<CTX, P> PrecompileProvider<CTX> for PrecompilesMap<CTX, P>
where
    CTX: ContextTr + Debug,
    CTX::Db: Database,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    type Output = InterpreterResult;

    fn set_spec(&mut self, _spec: <CTX::Cfg as Cfg>::Spec) -> bool {
        false
    }

    fn run(
        &mut self,
        context: &mut CTX,
        address: &Address,
        inputs: &InputsImpl,
        _is_static: bool,
        gas_limit: u64,
    ) -> Result<Option<Self::Output>, String> {
        // Check if address matches either static or dynamic precompiles
        if !self.contains(address) {
            return Ok(None);
        }

        let mut result = InterpreterResult {
            result: InstructionResult::Return,
            gas: Gas::new(gas_limit),
            output: Bytes::new(),
        };

        let result = if let Some(precompile) = self.dyn_precompiles.inner.get(address) {
            // === Dynamic precompile ===

            // Execute the precompile
            let input_bytes = match &inputs.input {
                CallInput::SharedBuffer(range) => {
                    #[allow(clippy::option_if_let_else)]
                    if let Some(slice) = context.local().shared_memory_buffer_slice(range.clone()) {
                        slice.to_vec()
                    } else {
                        vec![]
                    }
                }
                CallInput::Bytes(bytes) => bytes.to_vec(),
            };

            let precompile_result = precompile.call(PrecompileInput {
                data: &input_bytes,
                gas: gas_limit,
                caller: inputs.caller_address,
                value: inputs.call_value,
                internals: EvmInternals::new(context),
                target_address: inputs.target_address,
                bytecode_address: inputs.bytecode_address.expect("always set for precompile calls"),
            });

            match precompile_result {
                Ok(output) => {
                    let underflow = result.gas.record_cost(output.gas_used);
                    assert!(underflow, "Gas underflow is not possible");
                    result.result = if output.reverted {
                        InstructionResult::Revert
                    } else {
                        InstructionResult::Return
                    };
                    result.output = output.bytes;
                    result
                }
                Err(PrecompileError::Fatal(e)) => return Err(e),
                Err(e) => {
                    result.result = if e.is_oog() {
                        InstructionResult::PrecompileOOG
                    } else {
                        InstructionResult::PrecompileError
                    };
                    result
                }
            }
        } else if let Some(inner_result) =
            self.inner.run(context, address, inputs, _is_static, gas_limit)?
        {
            // === Inner provider ===
            inner_result
        } else {
            return Ok(None);
        };

        Ok(Some(result))
    }

    fn warm_addresses(&self) -> Box<impl Iterator<Item = Address>> {
        Box::new(self.inner.warm_addresses().chain(self.dyn_precompiles.addresses.iter().copied()))
    }

    fn contains(&self, address: &Address) -> bool {
        self.contains(address)
    }
}

/// A dynamic precompile implementation that can be modified at runtime.
#[derive(Clone)]
pub struct DynPrecompile(pub(crate) Arc<dyn Precompile + Send + Sync>);

impl DynPrecompile {
    /// Creates a new [`DynPrecompiles`] with the given closure.
    pub fn new<F>(id: PrecompileId, f: F) -> Self
    where
        F: Fn(PrecompileInput<'_>) -> PrecompileResult + Send + Sync + 'static,
    {
        Self(Arc::new((id, f)))
    }

    /// Creates a new [`DynPrecompiles`] with the given closure and [`Precompile::is_pure`]
    /// returning `false`.
    pub fn new_stateful<F>(id: PrecompileId, f: F) -> Self
    where
        F: Fn(PrecompileInput<'_>) -> PrecompileResult + Send + Sync + 'static,
    {
        Self(Arc::new(StatefulPrecompile((id, f))))
    }

    /// Flips [`Precompile::is_pure`] to `false`.
    pub fn stateful(self) -> Self {
        Self(Arc::new(StatefulPrecompile(self.0)))
    }
}

impl core::fmt::Debug for DynPrecompile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynPrecompile").finish()
    }
}

/// A mutable representation of precompiles that allows for runtime modification.
///
/// This structure stores dynamic precompiles that can be modified at runtime,
/// unlike the static `Precompiles` struct from revm.
#[derive(Clone, Default)]
pub struct DynPrecompiles {
    /// Precompiles
    inner: HashMap<Address, DynPrecompile>,
    /// Addresses of precompile
    addresses: HashSet<Address>,
}

impl DynPrecompiles {
    /// Consumes the type and returns an iterator over the addresses and the corresponding
    /// precompile.
    pub fn into_precompiles(self) -> impl Iterator<Item = (Address, DynPrecompile)> {
        self.inner.into_iter()
    }
}

impl core::fmt::Debug for DynPrecompiles {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynPrecompiles").field("addresses", &self.addresses).finish()
    }
}

/// Input for a precompile call.
#[derive(Debug)]
pub struct PrecompileInput<'a> {
    /// Input data bytes.
    pub data: &'a [u8],
    /// Gas limit.
    pub gas: u64,
    /// Caller address.
    pub caller: Address,
    /// Value sent with the call.
    pub value: U256,
    /// Target address of the call. Would be the same as `bytecode_address` unless it's a
    /// DELEGATECALL.
    pub target_address: Address,
    /// Bytecode address of the call.
    pub bytecode_address: Address,
    /// Various hooks for interacting with the EVM state.
    pub internals: EvmInternals<'a>,
}

impl<'a> PrecompileInput<'a> {
    /// Returns the calldata of the call.
    pub const fn data(&self) -> &[u8] {
        self.data
    }

    /// Returns the caller address of the call.
    pub const fn caller(&self) -> &Address {
        &self.caller
    }

    /// Returns the gas limit of the call.
    pub const fn gas(&self) -> u64 {
        self.gas
    }

    /// Returns the value of the call.
    pub const fn value(&self) -> &U256 {
        &self.value
    }

    /// Returns the target address of the call.
    pub const fn target_address(&self) -> &Address {
        &self.target_address
    }

    /// Returns the bytecode address of the call.
    pub const fn bytecode_address(&self) -> &Address {
        &self.bytecode_address
    }

    /// Returns whether the call is a direct call, i.e when precompile was called directly and not
    /// via a DELEGATECALL/CALLCODE.
    pub fn is_direct_call(&self) -> bool {
        self.target_address == self.bytecode_address
    }

    /// Returns the [`EvmInternals`].
    pub const fn internals(&self) -> &EvmInternals<'_> {
        &self.internals
    }

    /// Returns a mutable reference to the [`EvmInternals`].
    pub const fn internals_mut(&mut self) -> &mut EvmInternals<'a> {
        &mut self.internals
    }
}

/// Trait for implementing precompiled contracts.
#[auto_impl::auto_impl(&, Arc)]
pub trait Precompile {
    /// Returns precompile ID.
    fn precompile_id(&self) -> &PrecompileId;

    /// Execute the precompile with the given input data, gas limit, and caller address.
    fn call(&self, input: PrecompileInput<'_>) -> PrecompileResult;

    /// Returns whether the precompile is pure.
    ///
    /// A pure precompile has deterministic output based solely on its input.
    /// Non-pure precompiles may produce different outputs for the same input
    /// based on the current state or other external factors.
    ///
    /// # Default
    ///
    /// Returns `true` by default, indicating the precompile is pure
    /// and its results should be cached as this is what most of the precompiles are.
    ///
    /// # Examples
    ///
    /// Override this method to return `false` for non-deterministic precompiles:
    ///
    /// ```ignore
    /// impl Precompile for MyDeterministicPrecompile {
    ///     fn call(&self, input: PrecompileInput<'_>) -> PrecompileResult {
    ///         // non-deterministic computation dependent on state
    ///     }
    ///
    ///     fn is_pure(&self) -> bool {
    ///         false // This precompile might produce different output for the same input
    ///     }
    /// }
    /// ```
    fn is_pure(&self) -> bool {
        true
    }
}

impl<F> Precompile for (PrecompileId, F)
where
    F: Fn(PrecompileInput<'_>) -> PrecompileResult + Send + Sync,
{
    fn precompile_id(&self) -> &PrecompileId {
        &self.0
    }

    fn call(&self, input: PrecompileInput<'_>) -> PrecompileResult {
        self.1(input)
    }
}

impl<F> Precompile for (&PrecompileId, F)
where
    F: Fn(PrecompileInput<'_>) -> PrecompileResult + Send + Sync,
{
    fn precompile_id(&self) -> &PrecompileId {
        self.0
    }

    fn call(&self, input: PrecompileInput<'_>) -> PrecompileResult {
        self.1(input)
    }
}

impl Precompile for revm::precompile::Precompile {
    fn precompile_id(&self) -> &PrecompileId {
        self.id()
    }

    fn call(&self, input: PrecompileInput<'_>) -> PrecompileResult {
        self.precompile()(input.data, input.gas)
    }
}

impl<F> From<F> for DynPrecompile
where
    F: Fn(PrecompileInput<'_>) -> PrecompileResult + Send + Sync + 'static,
{
    fn from(f: F) -> Self {
        Self::new(PrecompileId::Custom("closure".into()), f)
    }
}

impl From<PrecompileFn> for DynPrecompile {
    fn from(f: PrecompileFn) -> Self {
        let p = move |input: PrecompileInput<'_>| f(input.data, input.gas);
        p.into()
    }
}

impl<F> From<(PrecompileId, F)> for DynPrecompile
where
    F: Fn(PrecompileInput<'_>) -> PrecompileResult + Send + Sync + 'static,
{
    fn from((id, f): (PrecompileId, F)) -> Self {
        Self(Arc::new((id, f)))
    }
}

impl From<(PrecompileId, PrecompileFn)> for DynPrecompile {
    fn from((id, f): (PrecompileId, PrecompileFn)) -> Self {
        let p = move |input: PrecompileInput<'_>| f(input.data, input.gas);
        (id, p).into()
    }
}

impl Precompile for DynPrecompile {
    fn precompile_id(&self) -> &PrecompileId {
        self.0.precompile_id()
    }

    fn call(&self, input: PrecompileInput<'_>) -> PrecompileResult {
        self.0.call(input)
    }

    fn is_pure(&self) -> bool {
        self.0.is_pure()
    }
}

struct StatefulPrecompile<P>(P);

impl<P: Precompile> Precompile for StatefulPrecompile<P> {
    fn precompile_id(&self) -> &PrecompileId {
        self.0.precompile_id()
    }

    fn call(&self, input: PrecompileInput<'_>) -> PrecompileResult {
        self.0.call(input)
    }

    fn is_pure(&self) -> bool {
        false
    }
}

/// Trait for dynamically resolving precompile contracts.
///
/// This trait allows for runtime resolution of precompiles that aren't known
/// at initialization time.
pub trait PrecompileLookup: Send + Sync {
    /// Looks up a precompile at the given address.
    ///
    /// Returns `Some(precompile)` if a precompile exists at the address,
    /// or `None` if no precompile is found.
    fn lookup(&self, address: &Address) -> Option<DynPrecompile>;
}

/// Implement PrecompileLookup for closure types
impl<F> PrecompileLookup for F
where
    F: Fn(&Address) -> Option<DynPrecompile> + Send + Sync,
{
    fn lookup(&self, address: &Address) -> Option<DynPrecompile> {
        self(address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eth::EthEvmContext;
    use alloy_primitives::{address, Bytes};
    use revm::{database::EmptyDB, handler::EthPrecompiles, precompile::PrecompileOutput};

    #[test]
    fn test_apply_precompile() {
        let eth_precompiles = EthPrecompiles::default();
        let mut precompiles = PrecompilesMap::from(eth_precompiles);

        let mut ctx = EthEvmContext::new(EmptyDB::default(), Default::default());

        // create a test input for the precompile (identity precompile)
        let identity_address = address!("0x0000000000000000000000000000000000000004");
        let test_input = Bytes::from_static(b"test data");
        let gas_limit = 1000;

        // using the dynamic precompiles interface
        assert!(precompiles.contains(&identity_address));

        let result = precompiles
            .run(
                &mut ctx,
                &identity_address,
                &InputsImpl {
                    target_address: identity_address,
                    bytecode_address: Some(identity_address),
                    caller_address: Address::ZERO,
                    input: CallInput::Bytes(test_input.clone()),
                    call_value: U256::ZERO,
                },
                true,
                gas_limit,
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.output, test_input, "Identity precompile should return the input data");

        // define a function to modify the precompile
        // this will change the identity precompile to always return a fixed value
        let constant_bytes = Bytes::from_static(b"constant value");

        // define a function to modify the precompile to always return a constant value
        precompiles.apply_precompile(&identity_address, move |_original_dyn| {
            // create a new DynPrecompile that always returns our constant
            Some(
                (|_input: PrecompileInput<'_>| -> PrecompileResult {
                    Ok(PrecompileOutput::new(10, Bytes::from_static(b"constant value")))
                })
                .into(),
            )
        });

        assert!(precompiles.contains(&identity_address));

        let result = precompiles
            .run(
                &mut ctx,
                &identity_address,
                &InputsImpl {
                    target_address: identity_address,
                    bytecode_address: Some(identity_address),
                    caller_address: Address::ZERO,
                    input: CallInput::Bytes(test_input),
                    call_value: U256::ZERO,
                },
                true,
                gas_limit,
            )
            .unwrap()
            .unwrap();

        assert_eq!(
            result.output, constant_bytes,
            "Modified precompile should return the constant value"
        );
    }
}
