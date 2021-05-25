# # EXAMPLE USAGE
# mxE11 <- exp(-5.5 + c(0,-2.9,-3,-2.9,-1.8,-1.7,-1.7,-1.6,-1.4,-.8,-.2,.3,1.0,1.5,1.9,2.4,2.8,3.4,3.8))
# mxE22 <- c(0.1, 0.2, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.5, 0.6, 0.7, 0.9, 0.9)
# mxE33 <- c(0.11, 0.2, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.5, 0.6, 0.7, 0.9, 0.9)
# 
# Rprof(tmp <- tempfile())
# PeriodLifeTable(mx=rep(c(mxE11,mxE22,mxE33), 200), age=rep(c(c(0,1), seq(5, 85, 5)),600),ax=rep(rep(NA,19),600),sex = 1, full.table=TRUE)
# Rprof()
# summaryRprof(tmp)

.KTExtension <- function(lx70) {
	# lx70: lx for 70 and older (should have 4 rows - one for each age group 
	# 70-74, 75-79, 80-84, 85+ and as many columns as years)
	# This function calculates the average number of years lived by 
	# those who die in each age group 70-74, 75-79, 80-84, 85+.
	# Returns a matrix of same dimensions as lx70.
	
	# For age groups >= 70, calculate hazard rate using the approximation
	# mu(x+1/2) ~ -log(1 - q) = -log(p) where p is the probability of 
	# survival to the next age group and equals l(x+5) / lx	
	mux <- (log(lx70[-nrow(lx70), , drop = FALSE]) - log(lx70)[-1, , drop = FALSE]) / 5
	
	# Calculate lx for 1-year age groups from 70 to 85. For 70, 75, 80, 85 
	# use known values. For the rest use interpolation l71 = l70 * exp(-mu70), 
	# l72 = l70 * exp(-2 * mu70),..., l84 = l80 * exp(-4 * mu80)
	lx70 <- rbind(
		x70[rep(seq(3), each = 5), , drop = FALSE] * exp(-seq(0, 4) * mux[rep(seq(3), each = 5), , drop = FALSE]), 
		lx70[4, , drop = FALSE]
	)
	
	# Calculate dx and qx for 1-year age groups using lx, for ages >= 70
	dx70 <- rbind(
		lx70[-nrow(lx70), , drop = FALSE] - lx70[-1, , drop = FALSE], 
		lx70[nrow(lx70), , drop = FALSE]
	)
	qx70 <- dx70 / lx70
		
	# Run regression on logit of probability of dying
	logitqx70 <- log(qx70 / (1 - qx70)) 
	logitqx70[nrow(logitqx70), ] <- NA # Not defined for 85+
	y <- as.vector(logitqx70)
	x <- rep(70:85 + .5, length.out = length(y))
	num.yr <- length(y) / 16
	yr <- as.factor(rep(seq_len(num.yr), each = 16))
	w <- as.vector(dx70)
	if (nlevels(yr) == 1) {
		mod <- lm(y ~ x, weights = x)
		logA <- mod$coefficients[1] # intercept
		B <- mod$coefficients[2] # slope
	} else {
		mod <- lm(y ~ 0 + yr + yr:x, weights = w)
		logA <- mod$coefficients[paste0("yr", seq(num.yr))] # intercepts
		B <- mod$coefficients[paste0("yr", seq(num.yr), ":x")] # slopes
	}
	
	# Calculate qx for age x >= 85
	logitqx85 <- t(logA + outer(B, (85:129 + .5))) # predict logit qx for age >= 85
	qx85 <- exp(logitqx85) / (1 + exp(logitqx85)) # invert logit transform
	
	# Calculate lx values for age x >= 85
	lx85 <- matrix(nrow = nrow(logitqx85), ncol = ncol(logitqx85)) 
	lx85[1, ] <- lx70[nrow(lx70), ] # last entry of vector holding lx70-85 
	for (k in seq(2, nrow(lx85))) 
		lx85[k, ] <- lx85[k - 1, ] * (1 - qx85[k - 1, ])
	
	# Calculate dx for age x >= 85
	dx85 <- lx85 * qx85
	
	# Join lx70 (holding lx for 1-year age groups from 70 to 85) with 
	# lx85 (holding lx for 1-year age groups from 85 to 129)
	# For the intersecting point, corresponding to age 85, we keep
	# the value in lx85, estimated via the method above
	lx70 <- rbind(lx70[-nrow(lx70), , drop = FALSE], lx85)  
	dx70 <- rbind(dx70[-nrow(dx70), , drop = FALSE], dx85)
	
	# Collapse back to 5-year age groups 70-74, 75-79, 80-84 and 
	# 85+, calculating average number of years lived by those who
	# die in each age group 
	# We assume that deaths occur at the midpoint of each 1-year 
	# age group so the number of years lived in the age group
	# 70-74 by someone who dies at age 73 is 3.5; similarly, the 
	# number of years lived in the age group 85+ by someone who dies
	# at age 100 is 15.5, etc.
	# yl is the years lived in the (current) age group at the time of death
	yl <- 70:129 - c(rep(c(70, 75, 80), each = 5), rep(85, length(85:129))) + 0.5
	
	# 5 year age group that each individual age from 70 to 129 belongs to
	x5y <- seq(70, 85, 5)[findInterval(70:129, seq(70, 85, 5))]
	
	ax70 <- as.vector(t(
		sapply(split(seq(nrow(dx70)), x5y),
			function(v) colSums(dx70[v, , drop = FALSE] * yl[v]) / 
									colSums(dx70[v, , drop = FALSE]))
	))
	ax70
}

.young_age_extension <- function(mx, sex) {
	# for age bins 0, 1-4 calculate the ax
	# mx: mortality rates for ages 0 (mx[1,]), 1-4 (mx[2,])
	# sex: 1 - male, 2 - female. different sexes have different ax (see p.48
	# 	   Preston et al, Demography: measuring and modeling population processes, 
	#      2001). "The lower the mortality, the more heavily will infant deaths be
	#	   concentrated at the earliest stages of infancy". The contingency table
	#      is based on Coale and Demeny (1983), who fitted a line to international
	#      and intertemporal data
	ax = matrix(NA, nrow=dim(mx)[1],ncol=dim(mx)[2]) # initiate ax
	if (sex == 1) {
		ax[1,] <- ifelse(mx[1,]>=0.107, 0.330, 0.045 + 2.684 * mx[1,])
		ax[2,] <- ifelse(mx[1,]>=0.107, 1.352, 1.651 - 2.816 * mx[1,])
	}

	else if (sex == 2) {
		ax[1,] <- ifelse(mx[1,]>=0.107, 0.350, 0.053 + 2.800 * mx[1,])
		ax[2,] <- ifelse(mx[1,]>=0.107, 1.361, 1.522 - 1.518 * mx[1,])
	}

	else stop("Sex must be 1 (male) or 2 (female)")
	ax <- c(ax) # put columns on top of each other into a vector
	return(ax)
}

PeriodLifeTable <- function(age, mx, ax, sex, check.conv = FALSE, full.table = FALSE) {
	# Generate a period life table for the supplied inputs
	# age: vector of age groups, 0, 5, 10, ..., 80, 85 (may be repeated
	#     if more than 1 years of data are available)
	# mx: mortality rates corresponding to ages in "age"
	# ax: average number of years lived by those who die in each age group
	#     NA values are replaced by 2.5. The ax for the open-ended age group
	#     85+ is calculated using the Kannisto-Thatcher method [Thatcher et al, 
	#     The survivor ratio method for estimating numbers at high ages, 
	#     Demographic Research (6) 2002]. For age groups 5-9 to 80-84 the ax 
	#     are calibrated using an iterative procedure described on p.47 of 
	#     Preston et al, Demography: measuring and modeling population processes, 
	#     2001.
	# sex: argument for the .young_age_extension() method
	# check.conv: If TRUE, it will test that the iterative procedure to estimate
	#     ax (see above) has converged
	if (length(age) != length(mx) | length(age) != length(ax)) 
		stop("All input vectors must be of same length")

	ax[is.na(ax)] <- 2.5 # halfway between 5 year age bins
	
	# CANNOT HAVE NA RATES

	# INITIALISE qx, px, lx, dx WITH CORRECT DIMENSIONS
	# Replace zero rates by a small number for numerical reasons
	mx[mx == 0] <- 1e-10
	# Probability of dying between age x and x+4
	qx <- 5 * mx / (1 + (5 - ax) * mx)
	# If probability of dying is >1, set it to "almost" 1
	qx[qx > 1] <- 0.99999
	qx[age == 85] <- 1 # by definition
	px <- 1 - qx # probability of surviving to next age group
	lx <- rep(NA, length(px))
	lx[age == 0] <- 100000
	for (k in seq(5, 85, 5)) 
		lx[age == k] <- lx[age == k - 5] * px[age == k - 5]
	dx <- lx * qx
	# set ax values for young and old ages
	ax[age <  5]  <- .young_age_extension(matrix(mx[age < 5], nrow = 2), sex = sex)
	ax[age >= 70] <- .KTExtension(matrix(lx[age >= 70], nrow = 4))
	
	# ITERATE TO FIND THE BEST AX VALUES AND UPDATE qx, px, lx, dx accordingly
	num.iter <- 4 # Number of iterations - see Preston et al. p.47
	iter.dat <- vector("list", num.iter + 1)
	iter.dat[[1]] <- list(ax = ax, qx = qx, lx = lx, dx = dx)
	for (i in seq(num.iter)) {
		ax.new <- ax
		for (k in seq(5, 80, 5)) 
			ax.new[age == k] <- (-5 / 24 * dx[age == k - 5] +
				2.5 * dx[age == k] + 5 / 24 * dx[age == k + 5]) / dx[age == k]
		ax.new[age <= 10 | age >= 70] <- ax[age <= 10 | age >= 70] # ignore the loop values above for ages <= 10 and >= 70
		ax <- ax.new
		qx <- 5 * mx / (1 + (5 - ax) * mx)
		# Need to recode qx for 0, 1-4 which have bin width n=1 and n=4
		qx[age == 0] <- 1 * mx[age == 0] / (1 + (1 - ax[age == 0]) * mx[age == 0]) # n = 1, 0
		qx[age == 1] <- 4 * mx[age == 1] / (1 + (4 - ax[age == 1]) * mx[age == 1]) # n = 4, 1-4
		qx[qx > 1] <- 0.99999
		qx[age == 85] <- 1
		px <- 1 - qx
		lx <- rep(NA, length(px))
		lx[age == 0] <- 100000
		lx[age == 1] <- lx[age == 0] * px[age == 0]
		lx[age == 5] <- lx[age == 1] * px[age == 1]
		for (k in seq(10, 85, 5)) # the rest have bin width 5
			lx[age == k] <- lx[age == k - 5] * px[age == k - 5]
		dx <- lx * qx
		# save result of current iteration
		iter.dat[[i + 1]] <- list(ax = ax, qx = qx, lx = lx, dx = dx)	
	}
	
	if (check.conv) {
		ax.iter <- sapply(iter.dat, `[[`, "ax")
		stopifnot(ax.iter[, num.iter] - ax.iter[, num.iter - 1] < 0.01)
	}
	iter.result <- iter.dat[[num.iter + 1]]
	ax <- iter.result$ax
	qx <- iter.result$qx
	lx <- iter.result$lx
	dx <- iter.result$dx
	
	# CALCULATE Lx and Tx, and hence ex
	Lx <- rep(NA, length(age))
	# Need to recode qx for 0, 1-4 which have bin width n=1 and n=4
	Lx[age == 0] <- 1 * lx[age == 1] + ax[age == 0] * dx[age == 0]
	Lx[age == 1] <- 4 * lx[age == 5] + ax[age == 1] * dx[age == 1]
	for (k in seq(5, 80, 5))
		Lx[age == k] <- 5 * lx[age == k + 5] + ax[age == k] * dx[age == k]
	Lx[age == 85] <- lx[age == 85] / mx[age == 85]
	Tx <- rep(NA, length(age)) 
	Tx[age == 85] <- Lx[age == 85]
	for (k in rev(seq(5, 80, 5)))
		Tx[age == k] <- Tx[age == k + 5] + Lx[age == k]
	Tx[age == 1] <- Tx[age == 5] + Lx[age == 1]
	Tx[age == 0] <- Tx[age == 1] + Lx[age == 0]
	ex <- Tx / lx
	
	if (full.table) return(data.frame(ax = ax, mx = mx, qx = qx, ex = ex, Tx = Tx, Lx = Lx, lx = lx, age = age))
	data.frame(mx = mx, qx = qx, ex = ex)
}
