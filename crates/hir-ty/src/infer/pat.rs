//! Type inference for patterns.

use std::iter::repeat_with;

use chalk_ir::Mutability;
use hir_def::{
    body::Body,
    expr::{BindingAnnotation, Expr, Literal, Pat, PatId},
    path::Path,
    type_ref::ConstScalar,
};
use hir_expand::name::Name;

use crate::{
    consteval::intern_const_scalar,
    infer::{BindingMode, Expectation, InferenceContext, TypeMismatch},
    lower::lower_to_chalk_mutability,
    static_lifetime, ConcreteConst, ConstValue, Interner, Substitution, Ty, TyBuilder, TyExt,
    TyKind,
};

use super::PatLike;

#[derive(Debug, Copy, Clone)]
enum AdjustMode {
    Pass,
    Reset,
    Peel,
}

impl<'a> InferenceContext<'a> {
    /// Infers type for tuple struct pattern or its corresponding assignee expression.
    ///
    /// Ellipses found in the original pattern or expression must be filtered out.
    pub(super) fn infer_tuple_struct_pat_like<T: PatLike>(
        &mut self,
        path: Option<&Path>,
        expected: &Ty,
        default_bm: T::BindingMode,
        id: T,
        ellipsis: Option<usize>,
        subs: &[T],
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path, true);
        let var_data = def.map(|it| it.variant_data(self.db.upcast()));
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }
        self.unify(&ty, expected);

        let substs =
            ty.as_adt().map(|(_, s)| s.clone()).unwrap_or_else(|| Substitution::empty(Interner));

        let field_tys = def.map(|it| self.db.field_types(it)).unwrap_or_default();
        let (pre, post) = match ellipsis {
            Some(idx) => subs.split_at(idx),
            None => (subs, &[][..]),
        };
        let post_idx_offset = field_tys.iter().count().saturating_sub(post.len());

        let pre_iter = pre.iter().enumerate();
        let post_iter = (post_idx_offset..).zip(post.iter());
        for (i, &subpat) in pre_iter.chain(post_iter) {
            let expected_ty = var_data
                .as_ref()
                .and_then(|d| d.field(&Name::new_tuple_field(i)))
                .map_or(self.err_ty(), |field| {
                    field_tys[field].clone().substitute(Interner, &substs)
                });
            let expected_ty = self.normalize_associated_types_in(expected_ty);
            T::infer(self, subpat, &expected_ty, default_bm);
        }

        ty
    }

    /// Infers type for record pattern or its corresponding assignee expression.
    pub(super) fn infer_record_pat_like<T: PatLike>(
        &mut self,
        path: Option<&Path>,
        expected: &Ty,
        default_bm: T::BindingMode,
        id: T,
        subs: impl Iterator<Item = (Name, T)>,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path, false);
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }

        self.unify(&ty, expected);

        let substs =
            ty.as_adt().map(|(_, s)| s.clone()).unwrap_or_else(|| Substitution::empty(Interner));

        let field_tys = def.map(|it| self.db.field_types(it)).unwrap_or_default();
        let var_data = def.map(|it| it.variant_data(self.db.upcast()));

        for (name, inner) in subs {
            let expected_ty = var_data
                .as_ref()
                .and_then(|it| it.field(&name))
                .map_or(self.err_ty(), |f| field_tys[f].clone().substitute(Interner, &substs));
            let expected_ty = self.normalize_associated_types_in(expected_ty);

            T::infer(self, inner, &expected_ty, default_bm);
        }

        ty
    }

    /// Infers type for tuple pattern or its corresponding assignee expression.
    ///
    /// Ellipses found in the original pattern or expression must be filtered out.
    pub(super) fn infer_tuple_pat_like<T: PatLike>(
        &mut self,
        expected: &Ty,
        default_bm: T::BindingMode,
        ellipsis: Option<usize>,
        subs: &[T],
    ) -> Ty {
        let expectations = match expected.as_tuple() {
            Some(parameters) => &*parameters.as_slice(Interner),
            _ => &[],
        };

        let ((pre, post), n_uncovered_patterns) = match ellipsis {
            Some(idx) => (subs.split_at(idx), expectations.len().saturating_sub(subs.len())),
            None => ((&subs[..], &[][..]), 0),
        };
        let mut expectations_iter = expectations
            .iter()
            .cloned()
            .map(|a| a.assert_ty_ref(Interner).clone())
            .chain(repeat_with(|| self.table.new_type_var()));

        let mut inner_tys = Vec::with_capacity(n_uncovered_patterns + subs.len());

        inner_tys.extend(expectations_iter.by_ref().take(n_uncovered_patterns + subs.len()));

        // Process pre
        for (ty, pat) in inner_tys.iter_mut().zip(pre) {
            *ty = T::infer(self, *pat, ty, default_bm);
        }

        // Process post
        for (ty, pat) in inner_tys.iter_mut().skip(pre.len() + n_uncovered_patterns).zip(post) {
            *ty = T::infer(self, *pat, ty, default_bm);
        }

        TyKind::Tuple(inner_tys.len(), Substitution::from_iter(Interner, inner_tys))
            .intern(Interner)
    }

    pub(super) fn infer_pat(&mut self, pat: PatId, expected: &Ty, default_bm: BindingMode) -> Ty {
        let expected = self.resolve_ty_shallow(expected);

        let adjust_mode = self.adjust_mode(&self.body, pat);
        let (expected, default_bm) = self.binding_mode(pat, expected, default_bm, adjust_mode);

        let ty = match &self.body[pat] {
            Pat::Tuple { args, ellipsis } => {
                self.infer_tuple_pat_like(&expected, default_bm, *ellipsis, args)
            }
            Pat::Or(pats) => {
                if let Some((first_pat, rest)) = pats.split_first() {
                    let ty = self.infer_pat(*first_pat, &expected, default_bm);
                    for pat in rest {
                        self.infer_pat(*pat, &expected, default_bm);
                    }
                    ty
                } else {
                    self.err_ty()
                }
            }
            Pat::Ref { pat, mutability } => {
                let mutability = lower_to_chalk_mutability(*mutability);
                let expectation = match expected.as_reference() {
                    Some((inner_ty, _lifetime, exp_mut)) => {
                        if mutability != exp_mut {
                            // FIXME: emit type error?
                        }
                        inner_ty.clone()
                    }
                    _ => self.result.standard_types.unknown.clone(),
                };
                let subty = self.infer_pat(*pat, &expectation, default_bm);
                TyKind::Ref(mutability, static_lifetime(), subty).intern(Interner)
            }
            Pat::TupleStruct { path: p, args: subpats, ellipsis } => self
                .infer_tuple_struct_pat_like(
                    p.as_deref(),
                    &expected,
                    default_bm,
                    pat,
                    *ellipsis,
                    subpats,
                ),
            Pat::Record { path: p, args: fields, ellipsis: _ } => {
                let subs = fields.iter().map(|f| (f.name.clone(), f.pat));
                self.infer_record_pat_like(p.as_deref(), &expected, default_bm, pat.into(), subs)
            }
            Pat::Path(path) => {
                // FIXME use correct resolver for the surrounding expression
                let resolver = self.resolver.clone();
                self.infer_path(&resolver, path, pat.into()).unwrap_or_else(|| self.err_ty())
            }
            Pat::Bind { mode, name: _, subpat } => {
                let mode = match *mode {
                    BindingAnnotation::Unannotated => default_bm,
                    mode => BindingMode::convert(mode),
                };
                self.result.pat_binding_modes.insert(pat, mode);

                let inner_ty = match subpat {
                    Some(subpat) => self.infer_pat(*subpat, &expected, default_bm),
                    None => expected.clone(),
                };
                let inner_ty = self.insert_type_vars_shallow(inner_ty);

                let bound_ty = match mode {
                    BindingMode::Ref(mutability) => {
                        TyKind::Ref(mutability, static_lifetime(), expected.clone())
                            .intern(Interner)
                    }
                    BindingMode::Move => expected.clone(),
                };
                let bound_ty = self.insert_type_vars_shallow(bound_ty);
                self.write_pat_ty(pat, bound_ty);
                return inner_ty;
            }
            Pat::Slice { prefix, slice, suffix } => {
                let elem_ty = match expected.kind(Interner) {
                    TyKind::Array(st, _) | TyKind::Slice(st) => st.clone(),
                    _ => self.err_ty(),
                };

                for &pat_id in prefix.iter().chain(suffix.iter()) {
                    self.infer_pat(pat_id, &elem_ty, default_bm);
                }

                if let &Some(slice_pat_id) = slice {
                    let rest_pat_ty = match expected.kind(Interner) {
                        TyKind::Array(_, length) => {
                            let len = match length.data(Interner).value {
                                ConstValue::Concrete(ConcreteConst {
                                    interned: ConstScalar::UInt(len),
                                }) => len.checked_sub((prefix.len() + suffix.len()) as u128),
                                _ => None,
                            };
                            TyKind::Array(
                                elem_ty.clone(),
                                intern_const_scalar(
                                    len.map_or(ConstScalar::Unknown, |len| ConstScalar::UInt(len)),
                                    TyBuilder::usize(),
                                ),
                            )
                        }
                        _ => TyKind::Slice(elem_ty.clone()),
                    }
                    .intern(Interner);
                    self.infer_pat(slice_pat_id, &rest_pat_ty, default_bm);
                }

                match expected.kind(Interner) {
                    TyKind::Array(_, const_) => TyKind::Array(elem_ty, const_.clone()),
                    _ => TyKind::Slice(elem_ty),
                }
                .intern(Interner)
            }
            Pat::Wild => expected.clone(),
            Pat::Range { start, end } => {
                let start_ty = self.infer_expr(*start, &Expectation::has_type(expected.clone()));
                self.infer_expr(*end, &Expectation::has_type(start_ty))
            }
            Pat::Lit(expr) => self.infer_expr(*expr, &Expectation::has_type(expected.clone())),
            Pat::Box { inner } => match self.resolve_boxed_box() {
                Some(box_adt) => {
                    let (inner_ty, alloc_ty) = match expected.as_adt() {
                        Some((adt, subst)) if adt == box_adt => (
                            subst.at(Interner, 0).assert_ty_ref(Interner).clone(),
                            subst.as_slice(Interner).get(1).and_then(|a| a.ty(Interner).cloned()),
                        ),
                        _ => (self.result.standard_types.unknown.clone(), None),
                    };

                    let inner_ty = self.infer_pat(*inner, &inner_ty, default_bm);
                    let mut b = TyBuilder::adt(self.db, box_adt).push(inner_ty);

                    if let Some(alloc_ty) = alloc_ty {
                        b = b.push(alloc_ty);
                    }
                    b.fill_with_defaults(self.db, || self.table.new_type_var()).build()
                }
                None => self.err_ty(),
            },
            Pat::ConstBlock(expr) => {
                self.infer_expr(*expr, &Expectation::has_type(expected.clone()))
            }
            Pat::Missing => self.err_ty(),
        };
        // use a new type variable if we got error type here
        let ty = self.insert_type_vars_shallow(ty);
        if !self.unify(&ty, &expected) {
            self.result
                .type_mismatches
                .insert(pat.into(), TypeMismatch { expected, actual: ty.clone() });
        }
        self.write_pat_ty(pat, ty.clone());
        ty
    }

    fn adjust_mode(&self, body: &Body, pat: PatId) -> AdjustMode {
        // When we perform destructuring assignment, we disable default match bindings, which are
        // unintuitive in this context.
        // if !pat.default_binding_modes {
        //     return AdjustMode::Reset;
        // }
        match body[pat] {
            // Type checking these product-like types successfully always require
            // that the expected type be of those types and not reference types.
            Pat::Record { .. }
            | Pat::TupleStruct { .. }
            | Pat::Tuple { .. }
            | Pat::Box { .. }
            | Pat::Range { .. }
            | Pat::Slice { .. } => AdjustMode::Peel,
            // String and byte-string literals result in types `&str` and `&[u8]` respectively.
            // All other literals result in non-reference types.
            Pat::Lit(expr)
                if matches!(
                    body[expr],
                    Expr::Literal(Literal::String(..) | Literal::ByteString(..))
                )
             => {
                AdjustMode::Pass
            }
            Pat::Lit(_) => AdjustMode::Peel,
            // FIXME: ConstBlock/Path might actually evaluate to ref, but inference is unimplemented.
            Pat::Path(..) => AdjustMode::Peel,
            Pat::ConstBlock(..) => AdjustMode::Peel,
            // When encountering a `& mut? pat` pattern, reset to "by value".
            // This is so that `x` and `y` here are by value, as they appear to be:
            //
            // ```
            // match &(&22, &44) {
            //   (&x, &y) => ...
            // }
            // ```
            //
            // See issue #46688.
            Pat::Ref { .. } => AdjustMode::Reset,
            // A `_` pattern works with any expected type, so there's no need to do anything.
            Pat::Wild
            // Bindings also work with whatever the expected type is,
            // and moreover if we peel references off, that will give us the wrong binding type.
            // Also, we can have a subpattern `binding @ pat`.
            // Each side of the `@` should be treated independently (like with OR-patterns).
            | Pat::Bind { .. }
            // An OR-pattern just propagates to each individual alternative.
            // This is maximally flexible, allowing e.g., `Some(mut x) | &Some(mut x)`.
            // In that example, `Some(mut x)` results in `Peel` whereas `&Some(mut x)` in `Reset`.
            | Pat::Or(_) | Pat::Missing => AdjustMode::Pass,
        }
    }

    fn binding_mode(
        &mut self,
        pat: PatId,
        expected: Ty,
        default_bm: BindingMode,
        adjust_mode: AdjustMode,
    ) -> (Ty, BindingMode) {
        match adjust_mode {
            AdjustMode::Pass => (expected, default_bm),
            AdjustMode::Reset => (expected, BindingMode::default()),
            AdjustMode::Peel => self.peel_off_references(pat, expected, default_bm),
        }
    }

    /// Peel off as many immediately nested `& mut?` from the expected type as possible
    /// and return the new expected type and binding default binding mode.
    /// The adjustments vector, if non-empty is stored in a table.
    fn peel_off_references<'ty>(
        &mut self,
        pat: PatId,
        mut expected: Ty,
        mut default_bm: BindingMode,
    ) -> (Ty, BindingMode) {
        let mut pat_adjustments = Vec::new();
        while let Some((inner, _lifetime, mutability)) = expected.as_reference() {
            pat_adjustments.push(expected.clone());
            expected = self.resolve_ty_shallow(inner);
            default_bm = BindingMode::Ref(match default_bm {
                BindingMode::Move | BindingMode::Ref(Mutability::Mut) => mutability,
                BindingMode::Ref(Mutability::Not) => Mutability::Not,
            })
        }

        if !pat_adjustments.is_empty() {
            pat_adjustments.shrink_to_fit();
            self.result.pat_adjustments.insert(pat, pat_adjustments);
        }

        (expected, default_bm)
    }
}
