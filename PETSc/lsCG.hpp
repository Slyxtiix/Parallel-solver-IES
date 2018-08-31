
PetscErrorCode CGLS_FUNCTION(Mat A, Vec b, Vec x, Vec residual, PetscReal tol, PetscInt *k);
PetscErrorCode QpkCGLS_FUNCTION(Mat *A, Vec b, Vec *x, PetscReal tol, PetscInt P, PetscInt n);

PetscErrorCode CGLS_FUNCTION(Mat A, Vec b, Vec x, Vec residual, PetscReal tol, PetscInt *k){
	 
	Vec            rr, ss, pp, qq,  Atresidual; 
	PetscReal      gamma, gamma1, alpha, beta;
	PetscReal      temp, normr;
	PetscScalar    neg_one = -1.0,one = 1.0, zero=0.0;
	
	VecDuplicate(x,&ss);
	VecDuplicate(x,&pp);
	VecDuplicate(x,&Atresidual);
	 
	VecDuplicate(b,&rr);
	VecDuplicate(b,&qq);
	
	
	VecSet(x, zero);
	
	VecCopy(b,rr);
	MatMult(A,b, ss);
	VecCopy(ss,pp);
	VecNorm(ss,NORM_2,&gamma);
	gamma=gamma*gamma;
	*k=0;
	MatMultTranspose(A,x, residual);
	VecAXPY(residual,neg_one,b);
	
	
	MatMult(A,residual, Atresidual);
	VecNorm(Atresidual,NORM_2,&normr);
	
	
	while(*k<3 && normr>tol){
		MatMultTranspose(A,pp, qq);
		VecNorm(qq,NORM_2,&temp);
		alpha=gamma/(temp*temp);
		VecAXPY(x,alpha,pp);
		temp=-1*alpha;
		VecAXPY(rr,temp,qq);
		MatMult(A,rr, ss);
		VecNorm(ss,NORM_2,&gamma1);
		gamma1=gamma1*gamma1;
		beta=gamma1/gamma;
		gamma=gamma1;
		VecAYPX(pp,beta,ss);
		
		MatMultTranspose(A,x, residual);
		VecAXPY(residual,neg_one,b);
		
		MatMult(A,residual, Atresidual);
		VecNorm(Atresidual,NORM_2,&normr);
		
		
		*k=*k+1;
		
	}
	/********************************************************************************/
	 
	 
	VecDestroy(&rr);
	VecDestroy(&ss);
	VecDestroy(&pp);
	VecDestroy(&qq);
	VecDestroy(&Atresidual);
	
	
	return 0;
 }
 
PetscErrorCode QpkCGLS_FUNCTION(Mat *A, Vec b, Vec *x, PetscReal tol, PetscInt P, PetscInt n){
	 PetscInt      i;
	 PetscInt 	   itnum;
	 Vec		   residual;
	 
	 VecCreate(PETSC_COMM_WORLD,&residual);
     VecSetSizes(residual,PETSC_DECIDE,n);
     VecSetFromOptions(residual);
	
	
	 for (i=0; i<P; i++){
		CGLS_FUNCTION(A[i], b, x[i], residual, tol, &itnum);
		VecCopy(residual, b);
		//PetscPrintf(PETSC_COMM_WORLD, "Iteration number is %d\n", itnum);
		
	 }
	 
	 for (i=P-2; i>=0; i--){
		CGLS_FUNCTION(A[i], b, x[i], residual, tol, &itnum);
		VecCopy(residual, b);
		//PetscPrintf(PETSC_COMM_WORLD, "Iteration number is %d\n", itnum);
		
	 }
	 
	 return 0;
}