// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Scoped thread-local storage
//!
//! This module provides the ability to generate *scoped* thread-local
//! variables. In this sense, scoped indicates that thread local storage
//! actually stores a reference to a value, and this reference is only placed
//! in storage for a scoped amount of time.
//!
//! There are no restrictions on what types can be placed into a scoped
//! variable, but all scoped variables are initialized to the equivalent of
//! null. Scoped thread local storage is useful when a value is present for a known
//! period of time and it is not required to relinquish ownership of the
//! contents.
//!
//! # Examples
//!
//! ## Basic usage
//!
//! ```
//! use scoped_tls_hkt::scoped_thread_local;
//!
//! scoped_thread_local!(static FOO: u32);
//!
//! # fn main() {
//! // Initially each scoped slot is empty.
//! assert!(!FOO.is_set());
//!
//! // When inserting a value, the value is only in place for the duration
//! // of the closure specified.
//! FOO.set(&1, || {
//!     FOO.with(|slot| {
//!         assert_eq!(*slot, 1);
//!     });
//! });
//! # }
//! ```
//!
//! ## Mutable value
//!
//! ```
//! use scoped_tls_hkt::scoped_thread_local;
//!
//! scoped_thread_local!(static mut FOO: u32);
//!
//! # fn main() {
//! // Initially each scoped slot is empty.
//! assert!(!FOO.is_set());
//!
//! // When inserting a value, the value is only in place for the duration
//! // of the closure specified.
//! let mut x = 1;
//! FOO.set(&mut x, || {
//!     FOO.with(|slot| {
//!         assert_eq!(*slot, 1);
//!
//!         // We can mutate the value
//!         *slot = 42;
//!     });
//! });
//!
//! // Changes will be visible externally
//! assert_eq!(x, 42);
//! # }
//! ```
//!
//! ## Higher-kinded types
//!
//! ```
//! use scoped_tls_hkt::scoped_thread_local;
//!
//! // Must implement Copy
//! #[derive(Copy, Clone)]
//! struct Foo<'a> {
//!     x: &'a str, // Lifetime is covariant
//!     y: i32,
//! }
//!
//! scoped_thread_local!(static FOO: for<'a> Foo<'a>);
//!
//! # fn main() {
//! // Initially each scoped slot is empty.
//! assert!(!FOO.is_set());
//!
//! // When inserting a value, the value is only in place for the duration
//! // of the closure specified.
//! FOO.set(Foo { x: "Hello", y: 42 }, || {
//!     FOO.with(|slot| {
//!         assert_eq!(slot.x, "Hello");
//!         assert_eq!(slot.y, 42);
//!     });
//! });
//! # }
//! ```
//!
//! ## Mutable higher-kinded types
//!
//! For mutable HKTs, the types must implement the [`ReborrowMut`](ReborrowMut)
//! trait, and the `Result` associated type should be the `Self` type, but with
//! the lifetime substituted with the trait's lifetime parameter.
//!
//! The [`ReborrowMut`](ReborrowMut) trait is implemented automatically for
//! many built-in types, including primitive types, references, mutable
//! references and tuples (up to length 10). Where this is insufficient, you
//! can implement the trait yourself: doing so should not require any unsafe
//! code.
//!
//! ```
//! use scoped_tls_hkt::scoped_thread_local;
//!
//! scoped_thread_local!(static mut FOO: for<'a> (&'a mut i32, &'a mut f32));
//!
//! # fn main() {
//! // Initially each scoped slot is empty.
//! assert!(!FOO.is_set());
//!
//! // References to local variables can be stored.
//! let mut x = 1;
//! let mut y = 2.0;
//! FOO.set((&mut x, &mut y), || {
//!     FOO.with(|(u, v)| {
//!         assert_eq!(*u, 1);
//!         assert_eq!(*v, 2.0);
//!         *u = 42;
//!     });
//! });
//!
//! assert_eq!(x, 42);
//! # }
//! ```

#![deny(missing_docs, warnings)]

use std::cell::Cell;
use std::thread::LocalKey;

/// Trait representing the act of "reborrowing" a mutable reference
/// to produce a new one with a shorter lifetime.
pub trait ReborrowMut<'a> {
    /// Type of the shorter reference
    type Result;

    /// Produces a new reference with lifetime 'a
    fn reborrow_mut(&'a mut self) -> Self::Result;
}

impl<'a, 'b: 'a, T: ?Sized> ReborrowMut<'a> for &'b mut T {
    type Result = &'a mut T;
    fn reborrow_mut(&'a mut self) -> Self::Result {
        &mut **self
    }
}

impl<'a, 'b: 'a, T: ?Sized> ReborrowMut<'a> for &'b T {
    type Result = &'a T;
    fn reborrow_mut(&'a mut self) -> Self::Result {
        &**self
    }
}

macro_rules! define_tuple_reborrow {
    (@expand $($t:ident),*) => {
        impl<'a, $($t,)*> ReborrowMut<'a> for ($($t,)*)
        where
            $($t: ReborrowMut<'a> + 'a),*
        {
            type Result = ($($t::Result,)*);
            #[allow(clippy::unused_unit)]
            fn reborrow_mut(&'a mut self) -> Self::Result {
                #[allow(non_snake_case)]
                let ($($t,)*) = self;
                ($($t.reborrow_mut(),)*)
            }
        }
    };
    () => {
        define_tuple_reborrow!(@expand);
    };
    ($t:ident $(, $ts:ident)*) => {
        define_tuple_reborrow!(@expand $t $(, $ts)*);
        define_tuple_reborrow!($($ts),*);
    };
}

macro_rules! define_copy_reborrow {
    ($($t:ty,)*) => {
        $(
            impl<'a> ReborrowMut<'a> for $t {
                type Result = $t;
                fn reborrow_mut(&'a mut self) -> Self::Result {
                    *self
                }
            }
        )*
    }
}

define_tuple_reborrow!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
define_copy_reborrow! {
    bool, char, isize, usize,
    i8, u8, i16, u16, i32, u32, i64, u64, i128, u128,
    f32, f64,
    std::any::TypeId,
}

/// The macro. See the module level documentation for the description and examples.
#[macro_export]
macro_rules! scoped_thread_local {
    ($(#[$attrs:meta])* $vis:vis static $name:ident: $(#[$tattrs:meta])* for<$lt:lifetime> $ty:ty) => (
        $(#[$tattrs])*
        #[allow(non_camel_case_types)]
        $vis struct $name<$lt> where ::std::cell::Cell<::std::option::Option<$ty>>: 'static {
            inner: &$lt ::std::thread::LocalKey<::std::cell::Cell<::std::option::Option<$ty>>>,
        }
        const _:() = {
            type Hkt<$lt> = $ty;
            use ::std::cell::Cell;
            use ::std::option::Option;
            use ::std::marker::Sync;
            use ::std::ops::{FnOnce, Drop};
            use ::std::thread::LocalKey;

            unsafe impl Sync for $name<'static> {}

            unsafe fn cast_to_static(x: Hkt<'_>) -> Hkt<'static> {
                std::mem::transmute(x)
            }

            // This wrapper helps to ensure that the 'static lifetime is not visible
            // to the safe code.
            fn cast_from_static<'a, 'b>(x: &'a Hkt<'static>) -> Hkt<'b> where 'a: 'b {
                *x
            }

            impl $name<'static> {
                pub fn set<F, R>(&'static self, t: Hkt<'_>, f: F) -> R
                    where F: FnOnce() -> R
                {
                    struct Reset {
                        key: &'static LocalKey<Cell<Option<Hkt<'static>>>>,
                        val: Option<Hkt<'static>>,
                    }
                    impl Drop for Reset {
                        fn drop(&mut self) {
                            self.key.with(|c| c.set(self.val.take()));
                        }
                    }
                    let prev = self.inner.with(|c| {
                        // Safety: we are only changing the lifetime. We enforce the
                        // lifetime constraints via the `Reset` struct.
                        c.replace(Some(unsafe { cast_to_static(t) }))
                    });
                    let _reset = Reset { key: self.inner, val: prev };
                    f()
                }

                #[track_caller]
                pub fn with<F, R>(&'static self, f: F) -> R
                    where F: FnOnce(Hkt<'_>) -> R
                {
                    let val = self.inner.with(|c| c.get());
                    let val = val.expect("cannot access a scoped thread local variable without calling `set` first");

                    // This also asserts that Hkt is covariant
                    f(cast_from_static(&val))
                }

                /// Test whether this TLS key has been `set` for the current thread.
                pub fn is_set(&'static self) -> bool {
                    self.inner.with(|c| c.get().is_some())
                }
            }
        };

        $(#[$attrs])*
        $vis static $name: $name<'static> = {
            type Hkt<$lt> = $ty;
            use ::std::cell::Cell;
            use ::std::option::Option;

            thread_local!(static FOO: Cell<Option<Hkt<'static>>> = const {
                Cell::new(None)
            });

            $name {
                inner: &FOO,
            }
        };
    );
    ($(#[$attrs:meta])* $vis:vis static mut $name:ident: $(#[$tattrs:meta])* for<$lt:lifetime> $ty:ty) => (
        $(#[$tattrs])*
        #[allow(non_camel_case_types)]
        $vis struct $name<$lt> where ::std::cell::Cell<::std::option::Option<$ty>>: 'static {
            inner: &$lt ::std::thread::LocalKey<::std::cell::Cell<::std::option::Option<$ty>>>,
        }
        const _:() = {
            type Hkt<$lt> = $ty;

            use ::std::cell::Cell;
            use ::std::option::Option;
            use ::std::marker::Sync;
            use ::std::ops::{FnOnce, Drop};
            use ::std::thread::LocalKey;

            use $crate::ReborrowMut;

            unsafe impl Sync for $name<'static> {}

            unsafe fn cast_to_static(x: Hkt<'_>) -> Hkt<'static> {
                std::mem::transmute(x)
            }

            // This wrapper helps to ensure that the 'static lifetime is not visible
            // to the safe code.
            fn cast_from_static<'is_reborrow_mut_general_enough, 'a, 'b>(x: &'a mut Hkt<'is_reborrow_mut_general_enough>) -> Hkt<'b> where 'a: 'b {
                //let y: &'b mut Hkt<'_> = unsafe { std::mem::transmute(x) };
                <Hkt<'is_reborrow_mut_general_enough> as ReborrowMut<'_>>::reborrow_mut(x)
            }

            impl $name<'static> {
                fn replace<F, R>(&'static self, value: Option<Hkt<'_>>, f: F) -> R
                    where F: FnOnce(Option<Hkt<'_>>) -> R
                {
                    struct Reset {
                        key: &'static LocalKey<Cell<Option<Hkt<'static>>>>,
                        val: Option<Hkt<'static>>,
                    }
                    impl Drop for Reset {
                        fn drop(&mut self) {
                            self.key.with(|c| c.set(self.val.take()));
                        }
                    }
                    let prev = self.inner.with(move |c| {
                        // Safety: we are only changing the lifetime. We enforce the
                        // lifetime constraints via the `Reset` struct.
                        c.replace(value.map(|x| unsafe { cast_to_static(x) }))
                    });
                    let mut reset = Reset { key: self.inner, val: prev };
                    f(reset.val.as_mut().map(cast_from_static))
                }

                /// Inserts a value into this scoped thread local storage slot for a
                /// duration of a closure.
                pub fn set<F, R>(&'static self, t: Hkt<'_>, f: F) -> R
                    where F: FnOnce() -> R
                {
                    self.replace(Some(t), |_| f())
                }

                /// Gets a value out of this scoped variable.
                ///
                /// This function takes a closure which receives the value of this
                /// variable. For the duration of the closure, the key will appear
                /// unset.
                ///
                /// # Panics
                ///
                /// This function will panic if `set` has not previously been called,
                /// or if the call is nested inside another (multiple mutable borrows
                /// of the same value are not allowed).
                ///
                #[track_caller]
                pub fn with<F, R>(&'static self, f: F) -> R
                    where F: FnOnce(Hkt<'_>) -> R
                {
                    self.replace(None, |val| val.map(f))
                        .expect("cannot access a scoped thread local variable without calling `set` first")
                }

                /// Test whether this TLS key has been `set` for the current thread.
                pub fn is_set(&'static self) -> bool {
                    self.replace(None, |prev| prev.is_some())
                }

            }
        };
        $(#[$attrs])*
        $vis static $name: $name<'static> = {
            type Hkt<$lt> = $ty;
            use ::std::cell::Cell;
            use ::std::option::Option;
            thread_local!(static FOO: Cell<Option<Hkt<'static>>> = const {
                Cell::new(None)
            });

            $name {
                inner: &FOO,
            }
        };
    );
    ($(#[$attrs:meta])* $vis:vis static $name:ident: $ty:ty) => (
        $(#[$attrs])*
        $vis static $name: $crate::ScopedKey<$ty> = $crate::ScopedKey {
            inner: {
                thread_local!(static FOO: ::std::cell::Cell<::std::option::Option<&'static $ty>> = const {
                    ::std::cell::Cell::new(None)
                });
                &FOO
            },
        };
    );
    ($(#[$attrs:meta])* $vis:vis static mut $name:ident: $ty:ty) => (
        $(#[$attrs])*
        $vis static $name: $crate::ScopedKeyMut<$ty> = $crate::ScopedKeyMut {
            inner: {
                thread_local!(static FOO: ::std::cell::Cell<::std::option::Option<&'static mut $ty>> = const {
                    ::std::cell::Cell::new(None)
                });
                &FOO
            },
        };
    );
}

/// Type representing a thread local storage key corresponding to a reference
/// to the type parameter `T`.
///
/// Keys are statically allocated and can contain a reference to an instance of
/// type `T` scoped to a particular lifetime. Keys provides two methods, `set`
/// and `with`, both of which currently use closures to control the scope of
/// their contents.
pub struct ScopedKey<T: ?Sized + 'static> {
    #[doc(hidden)]
    pub inner: &'static LocalKey<Cell<Option<&'static T>>>,
}

unsafe impl<T: ?Sized + 'static> Sync for ScopedKey<T> {}

unsafe fn cast_to_static<T: ?Sized + 'static>(x: &T) -> &'static T {
    std::mem::transmute(x)
}

// This wrapper helps to ensure that the 'static lifetime is not visible
// to the safe code.
fn cast_from_static<'a, 'b, T: ?Sized + 'static>(x: &'a &T) -> &'b T
where
    'a: 'b,
{
    x
}

impl<T: ?Sized + 'static> ScopedKey<T> {
    /// Inserts a value into this scoped thread local storage slot for a
    /// duration of a closure.
    ///
    /// While `cb` is running, the value `t` will be returned by `get` unless
    /// this function is called recursively inside of `cb`.
    ///
    /// Upon return, this function will restore the previous value, if any
    /// was available.
    ///
    /// # Examples
    ///
    /// ```
    /// use scoped_tls_hkt::scoped_thread_local;
    ///
    /// scoped_thread_local!(static FOO: u32);
    ///
    /// # fn main() {
    /// FOO.set(&100, || {
    ///     let val = FOO.with(|v| *v);
    ///     assert_eq!(val, 100);
    ///
    ///     // set can be called recursively
    ///     FOO.set(&101, || {
    ///         // ...
    ///     });
    ///
    ///     // Recursive calls restore the previous value.
    ///     let val = FOO.with(|v| *v);
    ///     assert_eq!(val, 100);
    /// });
    /// # }
    /// ```
    pub fn set<F, R>(&'static self, t: &T, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        struct Reset<T: ?Sized + 'static> {
            key: &'static LocalKey<Cell<Option<&'static T>>>,
            val: Option<&'static T>,
        }
        impl<T: ?Sized + 'static> Drop for Reset<T> {
            fn drop(&mut self) {
                self.key.with(|c| c.set(self.val));
            }
        }
        let prev = self.inner.with(|c| {
            // Safety: we are only changing the lifetime. We enforce the
            // lifetime constraints via the `Reset` struct.
            c.replace(Some(unsafe { cast_to_static(t) }))
        });
        let _reset = Reset {
            key: self.inner,
            val: prev,
        };
        f()
    }

    /// Gets a value out of this scoped variable.
    ///
    /// This function takes a closure which receives the value of this
    /// variable.
    ///
    /// # Panics
    ///
    /// This function will panic if `set` has not previously been called.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use scoped_tls_hkt::scoped_thread_local;
    ///
    /// scoped_thread_local!(static FOO: u32);
    ///
    /// # fn main() {
    /// FOO.with(|slot| {
    ///     // work with `slot`
    /// # drop(slot);
    /// });
    /// # }
    /// ```
    #[track_caller]
    pub fn with<F, R>(&'static self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let val = self
            .inner
            .with(|c| c.get())
            .expect("cannot access a scoped thread local variable without calling `set` first");
        f(cast_from_static(&val))
    }

    /// Test whether this TLS key has been `set` for the current thread.
    pub fn is_set(&'static self) -> bool {
        self.inner.with(|c| c.get().is_some())
    }
}

/// Type representing a thread local storage key corresponding to a mutable reference
/// to the type parameter `T`.
///
/// Keys are statically allocated and can contain a reference to an instance of
/// type `T` scoped to a particular lifetime. Keys provides two methods, `set`
/// and `with`, both of which currently use closures to control the scope of
/// their contents.
///
/// This differs from a `ScopedKey` because it provides access through a mutable
/// reference. As a result, when the `with(..)` method is used to access the value,
/// the key will appear unset whilst the closure is running. This is to prevent
/// the value being borrowed a second time.
pub struct ScopedKeyMut<T: ?Sized + 'static> {
    #[doc(hidden)]
    pub inner: &'static LocalKey<Cell<Option<&'static mut T>>>,
}

unsafe impl<T: ?Sized + 'static> Sync for ScopedKeyMut<T> {}

unsafe fn cast_to_static_mut<T: ?Sized + 'static>(x: &mut T) -> &'static mut T {
    std::mem::transmute(x)
}

// This wrapper helps to ensure that the 'static lifetime is not visible
// to the safe code.
fn cast_from_static_mut<'a, 'b, T: ?Sized + 'static>(x: &'a mut &mut T) -> &'b mut T
where
    'a: 'b,
{
    x
}

impl<T: ?Sized + 'static> ScopedKeyMut<T> {
    fn replace<F, R>(&'static self, t: Option<&mut T>, f: F) -> R
    where
        F: FnOnce(Option<&mut T>) -> R,
    {
        struct Reset<T: ?Sized + 'static> {
            key: &'static LocalKey<Cell<Option<&'static mut T>>>,
            val: Option<&'static mut T>,
        }
        impl<T: ?Sized + 'static> Drop for Reset<T> {
            fn drop(&mut self) {
                self.key.with(|c| c.set(self.val.take()));
            }
        }
        let prev = self.inner.with(move |c| {
            // Safety: we are only changing the lifetime. We enforce the
            // lifetime constraints via the `Reset` struct.
            c.replace(t.map(|x| unsafe { cast_to_static_mut(x) }))
        });
        let mut reset = Reset {
            key: self.inner,
            val: prev,
        };
        f(reset.val.as_mut().map(cast_from_static_mut))
    }

    /// Inserts a value into this scoped thread local storage slot for a
    /// duration of a closure.
    pub fn set<F, R>(&'static self, t: &mut T, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.replace(Some(t), |_| f())
    }

    /// Gets a value out of this scoped variable.
    ///
    /// This function takes a closure which receives the value of this
    /// variable. For the duration of the closure, the key will appear
    /// unset.
    ///
    /// # Panics
    ///
    /// This function will panic if `set` has not previously been called,
    /// or if the call is nested inside another (multiple mutable borrows
    /// of the same value are not allowed).
    ///
    #[track_caller]
    pub fn with<F, R>(&'static self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        self.replace(None, |val| val.map(f))
            .expect("cannot access a scoped thread local variable without calling `set` first")
    }

    /// Test whether this TLS key has been `set` for the current thread.
    pub fn is_set(&'static self) -> bool {
        self.replace(None, |prev| prev.is_some())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::panic;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{channel, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread;

    scoped_thread_local!(static FOO: u32);

    #[test]
    fn smoke() {
        scoped_thread_local!(static BAR: u32);

        assert!(!BAR.is_set());
        BAR.set(&1, || {
            assert!(BAR.is_set());
            BAR.with(|slot| {
                assert_eq!(*slot, 1);
            });
        });
        assert!(!BAR.is_set());
    }

    #[test]
    fn cell_allowed() {
        scoped_thread_local!(static BAR: Cell<u32>);

        BAR.set(&Cell::new(1), || {
            BAR.with(|slot| {
                assert_eq!(slot.get(), 1);
            });
        });
    }

    #[test]
    fn scope_item_allowed() {
        assert!(!FOO.is_set());
        FOO.set(&1, || {
            assert!(FOO.is_set());
            FOO.with(|slot| {
                assert_eq!(*slot, 1);
            });
        });
        assert!(!FOO.is_set());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn panic_resets() {
        struct Check(Sender<u32>);
        impl Drop for Check {
            fn drop(&mut self) {
                FOO.with(|r| {
                    self.0.send(*r).unwrap();
                })
            }
        }

        let (tx, rx) = channel();

        // Temporarily suppress panic output, as it would interfere
        // with the test harness output.
        let lock = PanicHookLock::acquire();
        let prev_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {
            // Do nothing
        }));

        let t = thread::spawn(|| {
            FOO.set(&1, || {
                let _r = Check(tx);

                FOO.set(&2, || panic!());
            });
        });

        let res = t.join();
        panic::set_hook(prev_hook);
        drop(lock);

        assert_eq!(rx.recv().unwrap(), 1);
        assert!(res.is_err());
    }

    /// Serializes tests that replace the process-wide panic hook.
    static PANIC_HOOK_LOCKED: AtomicBool = AtomicBool::new(false);

    struct PanicHookLock;

    impl PanicHookLock {
        fn acquire() -> Self {
            while PANIC_HOOK_LOCKED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                thread::yield_now();
            }
            PanicHookLock
        }
    }

    impl Drop for PanicHookLock {
        fn drop(&mut self) {
            PANIC_HOOK_LOCKED.store(false, Ordering::Release);
        }
    }

    /// Runs `f`, which must panic, and returns the file and line the panic
    /// was reported at.
    fn panic_location<F: FnOnce()>(f: F) -> (String, u32) {
        let location = Arc::new(Mutex::new(None));
        let lock = PanicHookLock::acquire();
        let prev_hook = panic::take_hook();
        {
            let location = location.clone();
            panic::set_hook(Box::new(move |info| {
                *location.lock().unwrap() =
                    info.location().map(|l| (l.file().to_string(), l.line()));
            }));
        }
        let res = panic::catch_unwind(panic::AssertUnwindSafe(f));
        panic::set_hook(prev_hook);
        drop(lock);

        assert!(res.is_err(), "expected a panic");
        let location = location.lock().unwrap().take();
        location.expect("panic hook did not record a location")
    }

    #[test]
    fn with_panics_at_caller() {
        scoped_thread_local!(static REF: u32);
        scoped_thread_local!(static mut MUT: u32);
        scoped_thread_local!(static HKT: for<'a> &'a u32);
        scoped_thread_local!(static mut HKT_MUT: for<'a> &'a mut u32);
        let here = |line| (file!().to_string(), line);

        // Never set.
        let (loc, line) = (panic_location(|| REF.with(|_| ())), line!());
        assert_eq!(loc, here(line));
        let (loc, line) = (panic_location(|| MUT.with(|_| ())), line!());
        assert_eq!(loc, here(line));
        let (loc, line) = (panic_location(|| HKT.with(|_| ())), line!());
        assert_eq!(loc, here(line));
        let (loc, line) = (panic_location(|| HKT_MUT.with(|_| ())), line!());
        assert_eq!(loc, here(line));

        // Set, then reset on leaving `set`.
        REF.set(&1, || ());
        assert!(!REF.is_set());
        let (loc, line) = (panic_location(|| REF.with(|_| ())), line!());
        assert_eq!(loc, here(line));
        HKT.set(&1, || ());
        assert!(!HKT.is_set());
        let (loc, line) = (panic_location(|| HKT.with(|_| ())), line!());
        assert_eq!(loc, here(line));

        // Nested mutable borrow.
        MUT.set(&mut 1, || {
            MUT.with(|_| {
                let (loc, line) = (panic_location(|| MUT.with(|_| ())), line!());
                assert_eq!(loc, here(line));
            })
        });
        assert!(!MUT.is_set());
        HKT_MUT.set(&mut 1, || {
            HKT_MUT.with(|_| {
                let (loc, line) = (panic_location(|| HKT_MUT.with(|_| ())), line!());
                assert_eq!(loc, here(line));
            })
        });
        assert!(!HKT_MUT.is_set());
    }

    #[test]
    fn attrs_allowed() {
        scoped_thread_local!(
            /// Docs
            static BAZ: u32
        );

        scoped_thread_local!(
            #[allow(non_upper_case_globals)]
            static quux: u32
        );

        let _ = BAZ;
        let _ = quux;
    }

    #[test]
    fn hkt_struct() {
        #[derive(Copy, Clone)]
        pub struct Foo<'a> {
            x: &'a str,
            y: &'a i32,
        }
        scoped_thread_local!(static BAR: for<'a> Foo<'a>);

        assert!(!BAR.is_set());
        BAR.set(Foo { x: "hi", y: &1 }, || {
            assert!(BAR.is_set());
            BAR.with(|slot| {
                assert_eq!(slot.x, "hi");
                assert_eq!(slot.y, &1);
            });
        });
        assert!(!BAR.is_set());
    }

    #[test]
    fn hkt_trait() {
        scoped_thread_local!(static BAR: for<'a> &'a dyn std::fmt::Display);

        assert!(!BAR.is_set());
        BAR.set(&"Hello", || {
            assert!(BAR.is_set());
            BAR.with(|slot| {
                assert_eq!(slot.to_string(), "Hello");
            });
            BAR.set(&42, || {
                assert!(BAR.is_set());
                BAR.with(|slot| {
                    assert_eq!(slot.to_string(), "42");
                });
            });
        });
        assert!(!BAR.is_set());
    }

    #[test]
    fn mut_value() {
        scoped_thread_local!(static mut BAR: i32);

        assert!(!BAR.is_set());
        let mut x = 0;

        BAR.set(&mut x, || {
            assert!(BAR.is_set());
            BAR.with(|slot| {
                assert!(!BAR.is_set());
                assert_eq!(*slot, 0);
                *slot = 42;
            });
            let mut y = 2;
            BAR.set(&mut y, || {
                assert!(BAR.is_set());
                BAR.with(|slot| {
                    assert_eq!(*slot, 2);
                    *slot = 15;
                });
            });
            assert_eq!(y, 15);
            assert!(BAR.is_set());
        });
        assert!(!BAR.is_set());
        assert_eq!(x, 42);
    }

    #[test]
    fn mut_trait() {
        scoped_thread_local!(static mut BAR: dyn std::io::Write);

        assert!(!BAR.is_set());
        let mut x = Vec::new();

        BAR.set(&mut x, || {
            assert!(BAR.is_set());
            BAR.with(|slot| {
                slot.write_all(&[1, 2, 3]).unwrap();
            });
        });
        assert!(!BAR.is_set());
        assert_eq!(x, [1, 2, 3]);
    }

    #[test]
    fn hkt_mut_tuple() {
        scoped_thread_local!(static mut BAR: for<'a> (&'a mut i32, &'a mut f32));

        let mut x = 1;
        let mut y = 2.0;

        assert!(!BAR.is_set());
        BAR.set((&mut x, &mut y), || {
            assert!(BAR.is_set());
            BAR.with(|(u, v)| {
                assert_eq!(*u, 1);
                assert_eq!(*v, 2.0);
                assert!(!BAR.is_set());
                *u = 3;
                *v = 4.0;
            });
        });
        assert!(!BAR.is_set());
        assert_eq!(x, 3);
        assert_eq!(y, 4.0);
    }

    #[test]
    fn hkt_mut_trait() {
        scoped_thread_local!(static mut BAR: for<'a> (&'a mut (dyn std::fmt::Display + 'static), &'a mut dyn std::any::Any));

        assert!(!BAR.is_set());
        let mut x = "Hello";
        let mut y = 42;
        BAR.set((&mut x, &mut y), || {
            assert!(BAR.is_set());
            BAR.with(|(u, _)| {
                assert_eq!(u.to_string(), "Hello");
            });
        });
        assert!(!BAR.is_set());
    }

    #[test]
    fn hkt_mut_newtype() {
        struct Foo<'a> {
            x: &'a mut (dyn std::fmt::Display + 'a),
            y: i32,
        }

        impl<'a, 'b> crate::ReborrowMut<'a> for Foo<'b> {
            type Result = Foo<'a>;
            fn reborrow_mut(&'a mut self) -> Self::Result {
                Foo {
                    x: self.x,
                    y: self.y,
                }
            }
        }

        scoped_thread_local!(static mut BAR: for<'a> Foo<'a>);

        assert!(!BAR.is_set());
        let mut x = "Hello";
        BAR.set(Foo { x: &mut x, y: 1 }, || {
            assert!(BAR.is_set());
            BAR.with(|foo| {
                assert_eq!(foo.x.to_string(), "Hello");
            });
        });
        assert!(!BAR.is_set());
    }
}
