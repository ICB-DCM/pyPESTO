#ifndef _amici_TPL_MODELNAME_h
#define _amici_TPL_MODELNAME_h
#include <cmath>
#include <memory>
#include "amici/defines.h"
#include <sundials/sundials_sparse.h> //SlsMat definition
#include "amici/solver_cvodes.h"
#include "amici/model_ode.h"

namespace amici {
class Solver;
}

/**
 * @brief Wrapper function to instantiate the linked Amici model without knowing the name at compile time.
 * @return
 */
extern void J_model_conversion_reaction(realtype *J, const realtype t, const realtype *x, const double *p, const double *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void JB_model_conversion_reaction(realtype *JB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx);
extern void JDiag_model_conversion_reaction(realtype *JDiag, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void JSparse_model_conversion_reaction(SlsMat JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void JSparseB_model_conversion_reaction(SlsMat JSparseB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx);
extern void Jv_model_conversion_reaction(realtype *Jv, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *v, const realtype *w, const realtype *dwdx);
extern void JvB_model_conversion_reaction(realtype *JvB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *vB, const realtype *w, const realtype *dwdx);
extern void Jy_model_conversion_reaction(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJydsigma_model_conversion_reaction(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJydy_model_conversion_reaction(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dwdp_model_conversion_reaction(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void dwdx_model_conversion_reaction(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void dxdotdp_model_conversion_reaction(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp);
extern void dydx_model_conversion_reaction(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void dydp_model_conversion_reaction(double *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip);
extern void dsigmaydp_model_conversion_reaction(double *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const int ip);
extern void qBdot_model_conversion_reaction(realtype *qBdot, const int ip, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdp);
extern void sigmay_model_conversion_reaction(double *sigmay, const realtype t, const realtype *p, const realtype *k);
extern void sxdot_model_conversion_reaction(realtype *sxdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *sx, const realtype *w, const realtype *dwdx, const realtype *J, const realtype *dxdotdp);
extern void w_model_conversion_reaction(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void x0_model_conversion_reaction(realtype *x0, const realtype t, const realtype *p, const realtype *k);
extern void xBdot_model_conversion_reaction(realtype *xBdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx);
extern void xdot_model_conversion_reaction(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void y_model_conversion_reaction(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);

/**
 * @brief AMICI-generated model subclass.
 */
class Model_model_conversion_reaction : public amici::Model_ODE {
public:
    /**
     * @brief Default constructor.
     */
    Model_model_conversion_reaction()
        : amici::Model_ODE(
              2, // nx
              2, // nxtrue
              2, // ny
              2, // nytrue
              0, // nz
              0, // nztrue
              0, // nevent
              1, // nobjective
              1, // nw
              2, // ndwddx
              2, // ndwdp
              4, // nnz
              2, // ubw
              2, // lbw
              amici::SecondOrderMode::none, // o2mode
              std::vector<realtype>{0.5, 0.3}, // dynamic parameters
              std::vector<realtype>{}, // fixedParameters
              std::vector<int>{}, // plist
              std::vector<realtype>(2,0.0), // idlist
              std::vector<int>{} // z2event
              )
    {}

    /**
     * @brief Clone this model instance.
     * @return A deep copy of this instance.
     */
    virtual amici::Model* clone() const override { return new Model_model_conversion_reaction(*this); }

    /** model specific implementation for fJ
     * @param J Matrix to which the Jacobian will be written
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJ(realtype *J, const realtype t, const realtype *x, const double *p, const double *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        J_model_conversion_reaction(J, t, x, p, k, h, w, dwdx);
    }

    /** model specific implementation for fJB
     * @param JB Matrix to which the Jacobian will be written
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param xB Vector with the adjoint states
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJB(realtype *JB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx) override {
        JB_model_conversion_reaction(JB, t, x, p, k, h, xB, w, dwdx);
    }

    /** model specific implementation for fJDiag
     * @param JDiag Matrix to which the Jacobian will be written
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJDiag(realtype *JDiag, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        JDiag_model_conversion_reaction(JDiag, t, x, p, k, h, w, dwdx);
    }

    /** model specific implementation for fJSparse
     * @param JSparse Matrix to which the Jacobian will be written
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJSparse(SlsMat JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        JSparse_model_conversion_reaction(JSparse, t, x, p, k, h, w, dwdx);
    }

    /** model specific implementation for fJSparseB
     * @param JSparseB Matrix to which the Jacobian will be written
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param xB Vector with the adjoint states
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJSparseB(SlsMat JSparseB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx) override {
        JSparseB_model_conversion_reaction(JSparseB, t, x, p, k, h, xB, w, dwdx);
    }

    /** model specific implementation of fJrz
     * @param nllh regularization for event measurements z
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     **/
    virtual void fJrz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    /** model specific implementation for fJv
     * @param Jv Matrix vector product of J with a vector v
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param v Vector with which the Jacobian is multiplied
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJv(realtype *Jv, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *v, const realtype *w, const realtype *dwdx) override {
        Jv_model_conversion_reaction(Jv, t, x, p, k, h, v, w, dwdx);
    }

    /** model specific implementation for fJvB
     * @param JvB Matrix vector product of JB with a vector v
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param xB Vector with the adjoint states
     * @param vB Vector with which the Jacobian is multiplied
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJvB(realtype *JvB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *vB, const realtype *w, const realtype *dwdx) override {
        JvB_model_conversion_reaction(JvB, t, x, p, k, h, xB, vB, w, dwdx);
    }

    /** model specific implementation of fJy
     * @param nllh negative log-likelihood for measurements y
     * @param iy output index
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint
     * @param sigmay measurement standard deviation at timepoint
     * @param my measurements at timepoint
     **/
    virtual void fJy(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        Jy_model_conversion_reaction(nllh, iy, p, k, y, sigmay, my);
    }

    /** model specific implementation of fJz
     * @param nllh negative log-likelihood for event measurements z
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     * @param mz event measurements at timepoint
     **/
    virtual void fJz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    /** model specific implementation of fdJrzdsigma
     * @param dJrzdsigma Sensitivity of event penalization Jrz w.r.t.
     * standard deviation sigmaz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param rz model root output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     **/
    virtual void fdJrzdsigma(double *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    /** model specific implementation of fdJrzdz
     * @param dJrzdz partial derivative of event penalization Jrz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param rz model root output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     **/
    virtual void fdJrzdz(double *dJrzdz, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    /** model specific implementation of fdJydsigma
     * @param dJydsigma Sensitivity of time-resolved measurement
     * negative log-likelihood Jy w.r.t. standard deviation sigmay
     * @param iy output index
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint
     * @param sigmay measurement standard deviation at timepoint
     * @param my measurement at timepoint
     **/
    virtual void fdJydsigma(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydsigma_model_conversion_reaction(dJydsigma, iy, p, k, y, sigmay, my);
    }

    /** model specific implementation of fdJydy
     * @param dJydy partial derivative of time-resolved measurement negative log-likelihood Jy
     * @param iy output index
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint
     * @param sigmay measurement standard deviation at timepoint
     * @param my measurement at timepoint
     **/
    virtual void fdJydy(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydy_model_conversion_reaction(dJydy, iy, p, k, y, sigmay, my);
    }

    /** model specific implementation of fdJzdsigma
     * @param dJzdsigma Sensitivity of event measurement
     * negative log-likelihood Jz w.r.t. standard deviation sigmaz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     * @param mz event measurement at timepoint
     **/
    virtual void fdJzdsigma(double *dJzdsigma, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    /** model specific implementation of fdJzdz
     * @param dJzdz partial derivative of event measurement negative log-likelihood Jz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     * @param mz event measurement at timepoint
     **/
    virtual void fdJzdz(double *dJzdz, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    /** model specific implementation of fdeltasx
     * @param deltaqB sensitivity update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param ip sensitivity index
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param xB adjoint state
     **/
    virtual void fdeltaqB(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
    }

    /** model specific implementation of fdeltasx
     * @param deltasx sensitivity update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param w repeating elements vector
     * @param ip sensitivity index
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param sx state sensitivity
     * @param stau event-time sensitivity
     **/
    virtual void fdeltasx(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau) override {
    }

    /** model specific implementation of fdeltax
     * @param deltax state update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     **/
    virtual void fdeltax(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old) override {
    }

    /** model specific implementation of fdeltaxB
     * @param deltaxB adjoint state update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param xB current adjoint state
     **/
    virtual void fdeltaxB(double *deltaxB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
    }

    /** model specific implementation of fdrzdp
     * @param drzdp partial derivative of root output rz w.r.t. model parameters p
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param ip parameter index w.r.t. which the derivative is requested
     **/
    virtual void fdrzdp(double *drzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    /** model specific implementation of fdrzdx
     * @param drzdx partial derivative of root output rz w.r.t. model states x
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     **/
    virtual void fdrzdx(double *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    /** model specific implementation of fsigmay
     * @param dsigmaydp partial derivative of standard deviation of measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     * @param ip sensitivity index
     **/
    virtual void fdsigmaydp(double *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
        dsigmaydp_model_conversion_reaction(dsigmaydp, t, p, k, ip);

    }

    /** model specific implementation of fsigmaz
     * @param dsigmazdp partial derivative of standard deviation of event measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     * @param ip sensitivity index
     **/
    virtual void fdsigmazdp(double *dsigmazdp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
    }

    /** model specific implementation of dwdp
     * @param dwdp Recurring terms in xdot, parameter derivative
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param w vector with helper variables
     */
    virtual void fdwdp(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        dwdp_model_conversion_reaction(dwdp, t, x, p, k, h, w);
    }

    /** model specific implementation of dwdx
     * @param dwdx Recurring terms in xdot, state derivative
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param w vector with helper variables
     */
    virtual void fdwdx(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        dwdx_model_conversion_reaction(dwdx, t, x, p, k, h, w);
    }

    /** model specific implementation of fdxdotdp
     * @param dxdotdp partial derivative xdot wrt p
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param ip parameter index
     * @param w vector with helper variables
     * @param dwdp derivative of w wrt p
     */
    virtual void fdxdotdp(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp) override {
        dxdotdp_model_conversion_reaction(dxdotdp, t, x, p, k, h, ip, w, dwdp);
    }

    /** model specific implementation of fdydx
     * @param dydx partial derivative of observables y w.r.t. model states x
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     **/
    virtual void fdydx(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        dydx_model_conversion_reaction(dydx, t, x, p, k, h);
    }

    /** model specific implementation of fdydp
     * @param dydp partial derivative of observables y w.r.t. model parameters p
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param ip parameter index w.r.t. which the derivative is requested
     **/
    virtual void fdydp(double *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
        dydp_model_conversion_reaction(dydp, t, x, p, k, h, ip);
    }

    /** model specific implementation of fdzdp
     * @param dzdp partial derivative of event-resolved output z w.r.t. model parameters p
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param ip parameter index w.r.t. which the derivative is requested
     **/
    virtual void fdzdp(double *dzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    /** model specific implementation of fdzdx
     * @param dzdx partial derivative of event-resolved output z w.r.t. model states x
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     **/
    virtual void fdzdx(double *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    /** model specific implementation for fqBdot
     * @param qBdot adjoint quadrature equation
     * @param ip sensitivity index
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param xB Vector with the adjoint states
     * @param w vector with helper variables
     * @param dwdp derivative of w wrt p
     **/
    virtual void fqBdot(realtype *qBdot, const int ip, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdp) override {
        qBdot_model_conversion_reaction(qBdot, ip, t, x, p, k, h, xB, w, dwdp);
    }

    /** model specific implementation for froot
     * @param root values of the trigger function
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     **/
    virtual void froot(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    /** model specific implementation of frz
     * @param rz value of root function at current timepoint (non-output events not included)
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     **/
    virtual void frz(double *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    /** model specific implementation of fsigmay
     * @param sigmay standard deviation of measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     **/
    virtual void fsigmay(double *sigmay, const realtype t, const realtype *p, const realtype *k) override {
        sigmay_model_conversion_reaction(sigmay, t, p, k);
    }

    /** model specific implementation of fsigmaz
     * @param sigmaz standard deviation of event measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     **/
    virtual void fsigmaz(double *sigmaz, const realtype t, const realtype *p, const realtype *k) override {
    }

    /** model specific implementation of fsrz
     * @param srz Sensitivity of rz, total derivative
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param sx current state sensitivity
     * @param h heavyside vector
     * @param ip sensitivity index
     **/
    virtual void fsrz(double *srz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
    }

    /** model specific implementation of fstau
     * @param stau total derivative of event timepoint
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param sx current state sensitivity
     * @param ip sensitivity index
     * @param ie event index
     **/
    virtual void fstau(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip, const int ie) override {
    }

    /** model specific implementation of fsx0
     * @param sx0 initial state sensitivities
     * @param t initial time
     * @param x0 initial state
     * @param p parameter vector
     * @param k constant vector
     * @param ip sensitivity index
     **/
    virtual void fsx0(realtype *sx0, const realtype t,const realtype *x0, const realtype *p, const realtype *k, const int ip) override {
    }

    /** model specific implementation of fsxdot
     * @param sxdot sensitivity rhs
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param ip parameter index
     * @param sx Vector with the state sensitivities
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     * @param J jacobian
     * @param dxdotdp parameter derivative of residual function
     */
    virtual void fsxdot(realtype *sxdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *sx, const realtype *w, const realtype *dwdx, const realtype *J, const realtype *dxdotdp) override {
        sxdot_model_conversion_reaction(sxdot, t, x, p, k, h, ip, sx, w, dwdx, J, dxdotdp);
    }

    /** model specific implementation of fsz
     * @param sz Sensitivity of rz, total derivative
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     * @param sx current state sensitivity
     * @param ip sensitivity index
     **/
    virtual void fsz(double *sz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
    }

    /** model specific implementation of fw
     * @param w Recurring terms in xdot
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     */
    virtual void fw(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        w_model_conversion_reaction(w, t, x, p, k, h);
    }

    /** model specific implementation of fx0
     * @param x0 initial state
     * @param t initial time
     * @param p parameter vector
     * @param k constant vector
     **/
    virtual void fx0(realtype *x0, const realtype t, const realtype *p, const realtype *k) override {
        x0_model_conversion_reaction(x0, t, p, k);
    }

    /** model specific implementation for fxBdot
     * @param xBdot adjoint residual function
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param xB Vector with the adjoint states
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fxBdot(realtype *xBdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx) override {
        xBdot_model_conversion_reaction(xBdot, t, x, p, k, h, xB, w, dwdx);
    }

    /** model specific implementation for fxdot
     * @param xdot residual function
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param w vector with helper variables
     **/
    virtual void fxdot(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        xdot_model_conversion_reaction(xdot, t, x, p, k, h, w);
    }

    /** model specific implementation of fy
     * @param y model output at current timepoint
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     **/
    virtual void fy(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        y_model_conversion_reaction(y, t, x, p, k, h);
    }

    /** model specific implementation of fz
     * @param z value of event output
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h heavyside vector
     **/
    virtual void fz(double *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }



    /**
     * @brief Get names of the model parameters
     * @return the names
     */
    virtual std::vector<std::string> getParameterNames() const override { return std::vector<std::string> {"k2",
"k1",}; }

    /**
     * @brief Get names of the model states
     * @return the names
     */
    virtual std::vector<std::string> getStateNames() const override { return std::vector<std::string> {"A",
"B",}; }

    /**
     * @brief Get names of the fixed model parameters
     * @return the names
     */
    virtual std::vector<std::string> getFixedParameterNames() const override { return std::vector<std::string> {}; }

    /**
     * @brief Get names of the observables
     * @return the names
     */
    virtual std::vector<std::string> getObservableNames() const override { return std::vector<std::string> {"x0",
"x1",}; }

    /**
     * @brief Get ids of the model parameters
     * @return the ids
     */
    virtual std::vector<std::string> getParameterIds() const override { return std::vector<std::string> {"R1_k2",
"R1_k1",}; }
    
    /**
     * @brief Get ids of the model states
     * @return the ids
     */
    virtual std::vector<std::string> getStateIds() const override { return std::vector<std::string> {"A",
"B",}; }
    
    /**
     * @brief Get ids of the fixed model parameters
     * @return the ids
     */
    virtual std::vector<std::string> getFixedParameterIds() const override { return std::vector<std::string> {}; }
    
    /**
     * @brief Get ids of the observables
     * @return the ids
     */
    virtual std::vector<std::string> getObservableIds() const override { return std::vector<std::string> {"y0",
"y1",}; }

};

#endif /* _amici_TPL_MODELNAME_h */
